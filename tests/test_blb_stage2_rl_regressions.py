import csv
import contextlib
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import torch


@contextlib.contextmanager
def _workspace_tempdir():
    root = Path(__file__).resolve().parents[1] / "tmp_tests"
    root.mkdir(exist_ok=True)
    path = root / f"case_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield str(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)


class BLBActionFinalEvalRegressionTests(unittest.TestCase):
    def test_resolve_base_action_accepts_numpy_arrays_without_truthiness(self):
        from Paean.blb_action_eval import BLBActionFinalEvaluationModule

        runner = BLBActionFinalEvaluationModule.__new__(BLBActionFinalEvaluationModule)
        action = np.arange(12, dtype=int)

        resolved = runner._resolve_base_action(
            {
                "blb_v3_best_action_vec": action,
                "best_action_vec": np.ones(12, dtype=int),
                "best_action": [2] * 12,
            }
        )

        self.assertTrue(np.array_equal(resolved, action))


class BLBPolicyWarmstartRegressionTests(unittest.TestCase):
    def test_preferred_action_bias_drives_deterministic_sample_to_baseline(self):
        from blb_stage2_rl.action_space import layer_dims, make_all_max_action_vector
        from blb_stage2_rl.policy import BLBStage2Policy

        preferred = make_all_max_action_vector(num_layers=2)
        policy = BLBStage2Policy(
            state_dim=7,
            num_layers=2,
            per_layer_dims=layer_dims(),
            first_input_levels=5,
            d_hidden=16,
            d_layer_emb=4,
        )

        policy.apply_preferred_action_bias(preferred, gain=50.0)
        state = torch.zeros(1, 7)
        action, _log_prob, _value = policy.sample_action(state, deterministic=True)

        self.assertEqual(action.squeeze(0).tolist(), preferred.tolist())


class BLBTraceWriterRegressionTests(unittest.TestCase):
    def test_trace_writer_appends_structured_rollout_rows(self):
        from blb_stage2_rl.persistence import append_blb_episode_trace_row

        with _workspace_tempdir() as td:
            trace_path = append_blb_episode_trace_row(
                td,
                {
                    "episode": 120,
                    "total_episodes": 240,
                    "ppo_update_count": 1,
                    "rollout_reward_mean": -273.0,
                    "rollout_reward_max": -273.0,
                    "best_reward": -273.0,
                    "priority1_count": 120,
                    "priority2_count": 0,
                    "priority3_count": 0,
                    "invalid_count": 0,
                    "entropy": 0.5,
                },
            )
            append_blb_episode_trace_row(
                td,
                {
                    "episode": 240,
                    "total_episodes": 240,
                    "ppo_update_count": 2,
                    "rollout_reward_mean": -10.0,
                    "rollout_reward_max": 0.0,
                    "best_reward": 0.0,
                    "priority1_count": 10,
                    "priority2_count": 0,
                    "priority3_count": 110,
                    "invalid_count": 1,
                    "entropy": 0.4,
                },
            )

            with Path(trace_path).open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["episode"], "120")
        self.assertEqual(rows[0]["priority1_count"], "120")
        self.assertEqual(rows[1]["best_reward"], "0.0")

    def test_trace_writer_persists_rollout_eval_diagnostics(self):
        from blb_stage2_rl.persistence import append_blb_episode_trace_row

        with _workspace_tempdir() as td:
            trace_path = append_blb_episode_trace_row(
                td,
                {
                    "episode": 120,
                    "total_episodes": 120,
                    "ppo_update_count": 1,
                    "rollout_reward_mean": -30.0,
                    "rollout_metric1_mean": 0.875,
                    "rollout_metric2_mean": 0.8125,
                    "rollout_loss_mean": 0.341,
                    "rollout_loss_std_mean": 0.002,
                    "apply_error_count": 1,
                    "eval_error_count": 0,
                    "last_error": "BLB apply failed: example",
                    "best_reward": -30.0,
                },
            )

            with Path(trace_path).open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        self.assertIn("rollout_metric1_mean", rows[0])
        self.assertEqual(rows[0]["rollout_metric1_mean"], "0.875")
        self.assertEqual(rows[0]["apply_error_count"], "1")
        self.assertIn("BLB apply failed", rows[0]["last_error"])


class BLBRewardRegressionTests(unittest.TestCase):
    def test_invalid_chain_is_not_masked_by_accuracy_violation(self):
        from blb_stage2_rl.reward import (
            BaselineCostStats,
            EpisodeMetrics,
            RewardWeights,
            compute_reward,
        )

        breakdown = compute_reward(
            EpisodeMetrics(metric1_mean=0.0, loss_mean=float("inf"), loss_std=float("inf")),
            type("Signals", (), {"any_invalid": True, "total_bits_sum": 0, "total_fusion_count": 0})(),
            action_avg_k=13.0,
            baseline=BaselineCostStats(metric1_mean=0.875),
            weights=RewardWeights(invalid_penalty=30.0),
            acc_threshold=0.865,
            stab_threshold=1.0,
            any_invalid=True,
        )

        self.assertTrue(breakdown.invalid)
        self.assertEqual(breakdown.priority, 3)
        self.assertEqual(breakdown.reward, -30.0)


class BLBOptimizerBaselineRegressionTests(unittest.TestCase):
    def test_bridge_baseline_evaluation_bypasses_cfg_derived_payload(self):
        from rescale_optimizer_bridge import RescaleOptimizerBridge

        class RecordingInvoker:
            def __init__(self):
                self.calls = []

            def __call__(self, config_name, payload):
                self.calls.append((config_name, payload))
                return {
                    "fusion_count": 0,
                    "valid": True,
                    "result": {
                        "valid": True,
                        "chain": {"total_bits": 123},
                        "invalid_chain": None,
                    },
                }

        invoker = RecordingInvoker()
        bridge = RescaleOptimizerBridge(invoker=invoker)

        out = bridge.evaluate_baseline(config_name="block2_mrpc_L7")

        self.assertEqual(invoker.calls, [("block2_mrpc", {})])
        self.assertEqual(out.config_name, "block2_mrpc_L7")
        self.assertTrue(out.valid)
        self.assertEqual(out.total_bits, 123)

    def test_env_all_max_action_uses_optimizer_baseline_scoring(self):
        from blb_stage2_rl.action_space import load_max_sfs, make_all_max_action_vector
        from blb_stage2_rl.env import BLBStage2Env, BLBStage2EnvConfig, ProbeBatch
        from blb_stage2_rl.reward import BaselineCostStats, RewardWeights
        from rescale_optimizer_bridge import RescaleOptimizerOutput

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(()))

            def forward(self, **_kwargs):
                logits = torch.tensor([[0.0, 10.0], [10.0, 0.0]])
                return type("Output", (), {"logits": logits})()

        class FakeHandler:
            def __getattr__(self, name):
                if name.startswith(("replace_", "restore_")):
                    return lambda *args, **kwargs: None
                raise AttributeError(name)

        class FakeRescaleBridge:
            def __init__(self):
                self.calls = []

            def evaluate_blocks(self, requests):
                self.calls.append("cfg-derived")
                return {
                    name: RescaleOptimizerOutput(
                        config_name=name,
                        fusion_count=0,
                        total_bits=0,
                        invalid_chain={"reason": "cfg-derived path should not run"},
                        valid=False,
                        raw={},
                    )
                    for name in requests
                }

            def evaluate_baseline_blocks(self, requests):
                self.calls.append("baseline")
                return {
                    name: RescaleOptimizerOutput(
                        config_name=name,
                        fusion_count=0,
                        total_bits=100,
                        invalid_chain=None,
                        valid=True,
                        raw={},
                    )
                    for name in requests
                }

        bridge = FakeRescaleBridge()
        probe = ProbeBatch(
            input_ids=torch.ones((2, 4), dtype=torch.long),
            attention_mask=torch.ones((2, 4), dtype=torch.long),
            labels=torch.tensor([1, 0], dtype=torch.long),
        )
        env = BLBStage2Env(
            handler=FakeHandler(),
            model=TinyModel(),
            probe_batches=[probe],
            rescale_bridge=bridge,
            baseline=BaselineCostStats(total_bits_sum=500, total_fusion_count=0, avg_k=13.0),
            reward_weights=RewardWeights(),
            acc_threshold=0.5,
            stab_threshold=10.0,
            max_sfs=load_max_sfs("mrpc"),
            num_layers=1,
            env_cfg=BLBStage2EnvConfig(profile="mrpc", num_trials_per_step=1),
        )

        _obs, reward, done, info = env.step(make_all_max_action_vector(num_layers=1))

        self.assertEqual(bridge.calls, ["baseline"])
        self.assertTrue(done)
        self.assertFalse(info["invalid"])
        self.assertFalse(info["reward_breakdown"].invalid)
        self.assertAlmostEqual(reward, 0.0)

    def test_real_mrpc_all_max_baseline_optimizer_outputs_are_valid(self):
        from blb_stage2_rl.action_space import (
            action_vector_to_cfgs,
            build_optimizer_requests,
            load_max_sfs,
            make_all_max_action_vector,
        )
        from rescale_optimizer_bridge import (
            InProcessInvoker,
            RescaleOptimizerBridge,
            aggregate_optimizer_signals,
        )

        gelu_degree = [1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1]
        attn_degree = [2, 2, 5, 5, 5, 2, 5, 2, 5, 5, 6, 2]
        decoded = action_vector_to_cfgs(
            make_all_max_action_vector(num_layers=12),
            load_max_sfs("mrpc"),
            num_layers=12,
            gelu_degree=gelu_degree,
            attn_degree=attn_degree,
        )
        requests = build_optimizer_requests("mrpc", decoded.cfgs_dict())
        bridge = RescaleOptimizerBridge(
            invoker=InProcessInvoker.from_profile(
                rescale_optimizer_root="Rescale_optimizer",
                profile="mrpc",
            )
        )

        bridge.evaluate_blocks(requests)
        outputs = bridge.evaluate_baseline_blocks(requests)
        signals = aggregate_optimizer_signals(outputs)

        self.assertFalse(signals.any_invalid, signals.invalid_chains)
        self.assertEqual(signals.valid_block_count, 60)
        self.assertEqual(signals.invalid_block_count, 0)


class BLBActionDescriptionRegressionTests(unittest.TestCase):
    def test_action_description_names_every_noise_point_and_truncation(self):
        from blb_stage2_rl.action_space import (
            describe_action_vector,
            load_max_sfs,
            make_all_max_action_vector,
        )

        action = make_all_max_action_vector(num_layers=1)
        desc = describe_action_vector(
            action,
            max_sfs=load_max_sfs("mrpc"),
            num_layers=1,
            gelu_degree=[4],
            attn_degree=[2],
            profile="mrpc",
        )
        records = desc["records"]

        self.assertEqual(desc["action_length"], len(action))
        self.assertEqual(desc["num_layers"], 1)
        self.assertGreaterEqual(len(records), len(action))

        first_input = [r for r in records if r["block"] == "first_input"][0]
        self.assertEqual(first_input["field"], "first_input_sf")
        self.assertEqual(first_input["value_type"], "scaling_factor")
        self.assertEqual(first_input["N"], 8192)

        truncations = [r for r in records if r["value_type"] == "truncation_k"]
        self.assertTrue(any(r["block"] == "block2" and r["value"] == 13 for r in truncations))
        self.assertTrue(all("location" in r and "operation" in r for r in records))
        self.assertTrue(any(r["field"] == "square_rescale_sf_0" and r["block"] == "block3" for r in records))


class BLBProbeSizingRegressionTests(unittest.TestCase):
    def test_probe_batch_count_covers_requested_probe_size(self):
        from blb_stage2_rl.runner import _effective_probe_batch_count

        class Ev:
            batch_size = 16
            stage2_probe_size = 256

        class Cfg:
            probe_batch_count = 4

        self.assertEqual(_effective_probe_batch_count(Ev(), Cfg()), 16)

    def test_explicit_probe_batch_count_override_still_works(self):
        from blb_stage2_rl.runner import _effective_probe_batch_count

        class Ev:
            batch_size = 16
            stage2_probe_size = 256
            blb_v3_probe_batch_count = 3

        class Cfg:
            probe_batch_count = 4

        self.assertEqual(_effective_probe_batch_count(Ev(), Cfg()), 3)


class BLBPersistencePathRegressionTests(unittest.TestCase):
    def test_blb_progress_stays_under_stage2_noise_progress(self):
        from blb_stage2_rl.runner import resolve_blb_persistence_dir

        class DummyEvaluator:
            pass

        with _workspace_tempdir() as td:
            ev = DummyEvaluator()
            ev.run_output_dir = td
            path = Path(resolve_blb_persistence_dir(ev))

        self.assertEqual(path.name, "progress")
        self.assertEqual(path.parent.name, "stage2_noise")


if __name__ == "__main__":
    unittest.main()
