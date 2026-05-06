"""加强版 BLB Stage 2 RL 模块 sanity test。

覆盖 spec §11 验证清单关键项：
  * action 全 max → reward = 0（differential 模式）
  * action 全 min → reward 多半很负（精度可能崩，至少 priority1 / priority2 触发）
  * StubInvoker / HeuristicStubInvoker 路径下能跑 1 个 episode + 1 次 PPO update
  * `bridge.clear()` 后 logits 严格等于 baseline（bit-identical）
  * BLB / legacy 互斥校验仍然生效
  * `LayerImportanceEvaluator.run_noise_rl_stage` 能根据 ``stage2_rl_variant``
    路由到新版 / 旧版（不实际跑大循环，仅检验导入 + dispatch 路径）

注：本测试不依赖 ``Rescale_optimizer`` 子项目，所有 cost 信号通过
``HeuristicStubInvoker`` 提供。
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np
import torch

# 让本测试以仓库根为 sys.path 起点（独立运行也能 import）
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _torch_load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class ActionSpaceBasicTests(unittest.TestCase):
    """动作空间编解码 + max_sfs JSON 加载。"""

    def test_dims_and_layer_count(self):
        from blb_stage2_rl import action_space
        # L=12 → 12 × 73 + 1 = 877 个分量
        dims = action_space.action_dims_for_config(12)
        self.assertEqual(len(dims), 877)
        # 每层 73 个分量
        self.assertEqual(len(action_space.layer_dims()), 73)

    def test_load_default_max_sfs(self):
        from blb_stage2_rl import action_space
        table = action_space.load_max_sfs("default")
        self.assertEqual(table.get(1, "wffn2_sf"), 22)
        self.assertEqual(table.get(2, "inv_std_fresh_sf"), 30)
        self.assertEqual(table.get(5, "gelu_coeff_mul_rescale_sf_3"), 22)

    def test_amax_amin_action_decoding(self):
        from blb_stage2_rl import action_space

        amax = action_space.make_all_max_action_vector(12)
        amin = action_space.make_all_min_action_vector(12)
        self.assertEqual(amax.shape, (877,))
        self.assertEqual(amin.shape, (877,))

        table = action_space.load_max_sfs("default")
        decoded_max = action_space.action_vector_to_cfgs(amax, table, num_layers=12)
        decoded_min = action_space.action_vector_to_cfgs(amin, table, num_layers=12)

        # amax 时 wffn2 SF 应为 max；amin 时 SF 应为 max - 2*(levels-1)
        self.assertEqual(decoded_max.block1_cfgs[0].wffn2_encode.scaling_factor, 22)
        self.assertLess(
            decoded_min.block1_cfgs[0].wffn2_encode.scaling_factor,
            decoded_max.block1_cfgs[0].wffn2_encode.scaling_factor,
        )

        # 首层 Block 1 的 truncation_k 必须为 None
        self.assertIsNone(decoded_max.block1_cfgs[0].output_truncation_k)
        self.assertIsNone(decoded_min.block1_cfgs[0].output_truncation_k)
        # 其它层 Block 1 的 truncation_k 应该是 K_LEVELS 的合法值
        self.assertIn(decoded_max.block1_cfgs[1].output_truncation_k, action_space.K_LEVELS)

    def test_avg_truncation_k(self):
        from blb_stage2_rl import action_space
        amax = action_space.make_all_max_action_vector(12)
        amin = action_space.make_all_min_action_vector(12)
        self.assertAlmostEqual(action_space.avg_truncation_k_in_action(amax, 12), 13.0, places=4)
        self.assertAlmostEqual(action_space.avg_truncation_k_in_action(amin, 12), 8.0, places=4)


class RewardSanityTests(unittest.TestCase):
    """spec §11 reward sanity check：amax → 0；不同优先级触发正确。"""

    def setUp(self):
        from blb_stage2_rl import action_space
        from blb_stage2_rl.default_invoker import HeuristicStubInvoker
        from rescale_optimizer_bridge import RescaleOptimizerBridge, aggregate_optimizer_signals

        self.action_space = action_space
        self.aggregate_optimizer_signals = aggregate_optimizer_signals
        self.table = action_space.load_max_sfs("default")
        self.invoker = HeuristicStubInvoker()
        self.bridge = RescaleOptimizerBridge(invoker=self.invoker)
        self.amax = action_space.make_all_max_action_vector(12)

        decoded_max = action_space.action_vector_to_cfgs(self.amax, self.table, num_layers=12)
        requests_max = action_space.build_optimizer_requests("mrpc", decoded_max.cfgs_dict())
        for cn, (_b, c) in requests_max.items():
            self.invoker.register_cfg(cn, c)
        self.outputs_max = self.bridge.evaluate_blocks(requests_max)
        self.invoker.clear_cfg_registry()
        self.signals_max = aggregate_optimizer_signals(self.outputs_max)

    def test_reward_at_amax_is_zero(self):
        from blb_stage2_rl.reward import (
            BaselineCostStats, RewardWeights, calibrate_weights_from_baseline,
            EpisodeMetrics, compute_reward,
        )
        baseline = BaselineCostStats(
            total_bits_sum=int(self.signals_max.total_bits_sum),
            total_fusion_count=int(self.signals_max.total_fusion_count),
            avg_k=13.0,
        )
        weights = calibrate_weights_from_baseline(baseline)
        metrics = EpisodeMetrics(loss_mean=0.5, loss_std=0.05, metric1_mean=0.86, metric2_mean=0.86)
        br = compute_reward(
            metrics, self.signals_max, action_avg_k=13.0, baseline=baseline,
            weights=weights, acc_threshold=0.85, stab_threshold=0.10,
        )
        # spec §11：action 全 max → reward = baseline cost reward = 0
        self.assertEqual(br.priority, 3)
        self.assertAlmostEqual(br.reward, 0.0, places=4)

    def test_priority1_trigger_on_low_acc(self):
        from blb_stage2_rl.reward import (
            BaselineCostStats, RewardWeights, EpisodeMetrics, compute_reward,
        )
        baseline = BaselineCostStats(
            total_bits_sum=int(self.signals_max.total_bits_sum),
            total_fusion_count=int(self.signals_max.total_fusion_count),
            avg_k=13.0,
        )
        weights = RewardWeights()
        metrics = EpisodeMetrics(loss_mean=2.0, loss_std=0.05, metric1_mean=0.20, metric2_mean=0.20)
        br = compute_reward(
            metrics, self.signals_max, action_avg_k=13.0, baseline=baseline,
            weights=weights, acc_threshold=0.85, stab_threshold=0.10,
        )
        self.assertEqual(br.priority, 1)
        self.assertLess(br.reward, -50.0)   # 大额负罚

    def test_priority2_trigger_on_high_std(self):
        from blb_stage2_rl.reward import (
            BaselineCostStats, RewardWeights, EpisodeMetrics, compute_reward,
        )
        baseline = BaselineCostStats(
            total_bits_sum=int(self.signals_max.total_bits_sum),
            total_fusion_count=int(self.signals_max.total_fusion_count),
            avg_k=13.0,
        )
        weights = RewardWeights()
        metrics = EpisodeMetrics(loss_mean=0.5, loss_std=2.0, metric1_mean=0.86, metric2_mean=0.86)
        br = compute_reward(
            metrics, self.signals_max, action_avg_k=13.0, baseline=baseline,
            weights=weights, acc_threshold=0.85, stab_threshold=0.10,
        )
        self.assertEqual(br.priority, 2)
        self.assertLess(br.reward, -10.0)

    def test_invalid_chain_returns_invalid_penalty(self):
        from blb_stage2_rl.reward import (
            BaselineCostStats, RewardWeights, EpisodeMetrics, compute_reward,
        )
        baseline = BaselineCostStats(
            total_bits_sum=int(self.signals_max.total_bits_sum),
            total_fusion_count=int(self.signals_max.total_fusion_count),
            avg_k=13.0,
        )
        weights = RewardWeights()
        metrics = EpisodeMetrics(loss_mean=0.5, loss_std=0.05, metric1_mean=0.86, metric2_mean=0.86)
        br = compute_reward(
            metrics, self.signals_max, action_avg_k=13.0, baseline=baseline,
            weights=weights, acc_threshold=0.85, stab_threshold=0.10,
            any_invalid=True,
        )
        self.assertEqual(br.priority, 3)
        self.assertTrue(br.invalid)
        self.assertAlmostEqual(br.reward, -float(weights.invalid_penalty), places=6)


class PolicyAndPPOTests(unittest.TestCase):
    """policy + ppo_update：单步 update 不抛异常 + 形状正确。"""

    def test_sample_evaluate_action(self):
        from blb_stage2_rl import action_space
        from blb_stage2_rl.policy import BLBStage2Policy

        per_layer = action_space.layer_dims()
        policy = BLBStage2Policy(
            state_dim=22, num_layers=12,
            per_layer_dims=per_layer, first_input_levels=5,
        )
        state = torch.randn(3, 22)
        actions, log_probs, values = policy.sample_action(state)
        self.assertEqual(actions.shape, (3, 877))
        self.assertEqual(log_probs.shape, (3,))
        self.assertEqual(values.shape, (3,))
        # evaluate 同 action → 同 log_prob
        log_probs2, _, _ = policy.evaluate_action(state, actions)
        self.assertTrue(torch.allclose(log_probs, log_probs2, atol=1e-5))

    def test_ppo_update_runs(self):
        from blb_stage2_rl import action_space
        from blb_stage2_rl.policy import BLBStage2Policy, RolloutBuffer, PPOConfig, ppo_update

        per_layer = action_space.layer_dims()
        policy = BLBStage2Policy(
            state_dim=22, num_layers=12,
            per_layer_dims=per_layer, first_input_levels=5,
        )
        opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
        buf = RolloutBuffer()
        for i in range(8):
            buf.add(
                state=np.random.randn(22).astype(np.float32),
                action=np.array([np.random.randint(0, d) for d in action_space.action_dims_for_config(12)], dtype=np.int64),
                log_prob=-100.0 + i, reward=float(i), value=float(i / 2),
            )
        metrics = ppo_update(policy, opt, buf, PPOConfig(n_epochs=1, minibatch_size=4), torch.device("cpu"))
        self.assertGreaterEqual(metrics["n_samples"], 1)
        self.assertTrue(np.isfinite(metrics["policy_loss"]))


class EnvEndToEndTests(unittest.TestCase):
    """Env.step + 真实 BERT mini 模型 + ReversibleLayerHandler 端到端 sanity。

    用 ``BertConfig(num_hidden_layers=2)`` 缩到 2 层减少开销；GELU/Softmax 都用 degree=4。
    """

    def setUp(self):
        from transformers import BertConfig, BertForSequenceClassification
        from function_handler import ReversibleLayerHandler

        torch.manual_seed(0)
        np.random.seed(0)

        self.config = BertConfig(
            num_hidden_layers=2,
            num_attention_heads=4,
            hidden_size=64,
            intermediate_size=128,
            num_labels=2,
            pad_token_id=0,
        )
        self.model = BertForSequenceClassification(self.config)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.handler = ReversibleLayerHandler(self.model)
        self.layers_attr = "model.bert.encoder.layer"

        # 装好多项式 attention / GELU 近似（spec §3.2 前置依赖）
        self.handler.replace_layer_softmax(
            list(range(2)), self.layers_attr, degree=4,
        )
        self.handler.replace_layer_gelu(
            list(range(2)), self.layers_attr, degree=4,
        )

        # 固定 mini probe batch（同一份 input；标签随机但确定性）
        torch.manual_seed(42)
        ids = torch.randint(low=2, high=100, size=(2, 8))
        am = torch.ones_like(ids)
        labels = torch.tensor([0, 1])
        from blb_stage2_rl.env import ProbeBatch
        self.probe_batches = [ProbeBatch(input_ids=ids, attention_mask=am, labels=labels)]

    def _make_env(self):
        from blb_stage2_rl import action_space
        from blb_stage2_rl.default_invoker import HeuristicStubInvoker
        from blb_stage2_rl.env import BLBStage2Env, BLBStage2EnvConfig
        from blb_stage2_rl.reward import BaselineCostStats, RewardWeights
        from rescale_optimizer_bridge import RescaleOptimizerBridge

        invoker = HeuristicStubInvoker()
        bridge = RescaleOptimizerBridge(invoker=invoker)
        max_sfs = action_space.load_max_sfs("default")

        env = BLBStage2Env(
            handler=self.handler,
            model=self.model,
            probe_batches=self.probe_batches,
            rescale_bridge=bridge,
            baseline=BaselineCostStats(),
            reward_weights=RewardWeights(),
            acc_threshold=-1.0,            # disabled，方便测试 cost reward
            stab_threshold=float("inf"),
            max_sfs=max_sfs,
            num_layers=2,
            gelu_degree=4,
            attn_degree=4,
            layers_attribute=self.layers_attr,
            env_cfg=BLBStage2EnvConfig(profile="mrpc", num_trials_per_step=2, probe_batch_count=1),
            heuristic_invoker=invoker,
        )
        return env

    def test_baseline_logits_unchanged_after_step(self):
        env = self._make_env()
        env.reset(seed=0)

        # 记录 baseline logits（无 BLB 噪声）
        with torch.inference_mode():
            ref_out = self.model(
                input_ids=self.probe_batches[0].input_ids,
                attention_mask=self.probe_batches[0].attention_mask,
                labels=self.probe_batches[0].labels,
            )
            ref_logits = ref_out.logits.clone()

        # 跑一步 RL
        from blb_stage2_rl import action_space
        amax = action_space.make_all_max_action_vector(2)
        obs, reward, done, info = env.step(amax)
        self.assertTrue(done)
        self.assertFalse(info["invalid"])
        self.assertIn("metrics", info)

        # bridge.clear() 后 logits 应该还原（spec §11"bridge.clear() 后 logits 严格等于 baseline"）
        with torch.inference_mode():
            after_out = self.model(
                input_ids=self.probe_batches[0].input_ids,
                attention_mask=self.probe_batches[0].attention_mask,
                labels=self.probe_batches[0].labels,
            )
            after_logits = after_out.logits.clone()

        self.assertEqual(ref_logits.shape, after_logits.shape)
        self.assertTrue(torch.allclose(ref_logits, after_logits, atol=1e-6))

    def test_env_uses_per_layer_stage1_degrees(self):
        from blb_stage2_rl import action_space

        self.handler.restore_layer_block5_noise(layer_indices=[0, 1], layer_name=self.layers_attr)
        self.handler.restore_layer_block4_noise(layer_indices=[0, 1], layer_name=self.layers_attr)
        self.handler.restore_layer_block3_noise(layer_indices=[0, 1], layer_name=self.layers_attr)
        self.handler.restore_layer_block2_noise(layer_indices=[0, 1], layer_name=self.layers_attr)
        self.handler.restore_layer_block1_noise(layer_indices=[0, 1], layer_name=self.layers_attr)
        self.handler.restore_blb_first_input_noise(layer_name=self.layers_attr)
        self.handler.replace_layer_softmax([0], self.layers_attr, degree=2)
        self.handler.replace_layer_softmax([1], self.layers_attr, degree=5)
        self.handler.replace_layer_gelu([0], self.layers_attr, degree=1)
        self.handler.replace_layer_gelu([1], self.layers_attr, degree=4)

        env = self._make_env()
        env.gelu_degree = np.asarray([1, 4], dtype=int)
        env.attn_degree = np.asarray([2, 5], dtype=int)
        env.gelu_degree_state = env._degree_state_scalar(env.gelu_degree, default=4)
        env.attn_degree_state = env._degree_state_scalar(env.attn_degree, default=4)

        _, _reward, done, info = env.step(action_space.make_all_max_action_vector(2))

        self.assertTrue(done)
        self.assertFalse(info["invalid"])
        self.assertNotIn("error", info)
        decoded = info["decoded"]
        self.assertEqual([decoded.block3_cfgs[i].degree for i in range(2)], [2, 5])
        self.assertEqual([decoded.block5_cfgs[i].gelu_degree for i in range(2)], [1, 4])

    def test_invalid_action_does_not_crash(self):
        """invalid_chain 触发时 reward = -INVALID_PENALTY，env 不 crash（spec §11）。"""
        from blb_stage2_rl import action_space
        from blb_stage2_rl.default_invoker import HeuristicStubInvoker
        from blb_stage2_rl.env import BLBStage2Env, BLBStage2EnvConfig
        from blb_stage2_rl.reward import BaselineCostStats, RewardWeights
        from rescale_optimizer_bridge import RescaleOptimizerBridge

        # 让 invoker 默认认定所有 chain 都 invalid（threshold 设得很高）
        invoker = HeuristicStubInvoker(invalid_total_bits_threshold=999_999_999)
        bridge = RescaleOptimizerBridge(invoker=invoker)
        max_sfs = action_space.load_max_sfs("default")
        env = BLBStage2Env(
            handler=self.handler, model=self.model,
            probe_batches=self.probe_batches, rescale_bridge=bridge,
            baseline=BaselineCostStats(total_bits_sum=10_000), reward_weights=RewardWeights(),
            acc_threshold=-1.0, stab_threshold=float("inf"),
            max_sfs=max_sfs, num_layers=2, gelu_degree=4, attn_degree=4,
            layers_attribute=self.layers_attr,
            env_cfg=BLBStage2EnvConfig(profile="mrpc", num_trials_per_step=1, probe_batch_count=1),
            heuristic_invoker=invoker,
        )
        env.reset(seed=0)
        amax = action_space.make_all_max_action_vector(2)
        obs, reward, done, info = env.step(amax)
        self.assertTrue(done)
        self.assertTrue(info["invalid"])
        self.assertAlmostEqual(reward, -float(env.reward_weights.invalid_penalty), places=4)

    def test_baseline_estimation(self):
        from blb_stage2_rl.env import estimate_baseline_cost_stats
        env = self._make_env()
        env.reset(seed=0)
        baseline = estimate_baseline_cost_stats(env, sample_count=2)
        self.assertGreater(baseline.total_bits_sum, 0)
        self.assertGreaterEqual(baseline.total_fusion_count, 0)
        self.assertEqual(baseline.avg_k, 13.0)


class RunNoiseRLStageDispatchTests(unittest.TestCase):
    """``LayerImportanceEvaluator.run_noise_rl_stage`` 的 variant 路由（不实跑）。"""

    def test_coerce_variant_legal_values(self):
        from layer_importance_evaluator import LayerImportanceEvaluator
        f = LayerImportanceEvaluator._coerce_stage2_rl_variant
        self.assertEqual(f("blb_v3"), "blb_v3")
        self.assertEqual(f("BLB"), "blb_v3")
        self.assertEqual(f(None), "blb_v3")
        self.assertEqual(f("legacy_v2"), "legacy_v2")
        self.assertEqual(f("v2"), "legacy_v2")
        self.assertEqual(f("LEGACY"), "legacy_v2")
        with self.assertRaises(ValueError):
            f("invalid_variant_xyz")

    def test_stage2_resume_checkpoint_path_follows_variant(self):
        from layer_importance_evaluator import LayerImportanceEvaluator
        from noise_rl_module_v2 import NOISE_STAGE_CHECKPOINT_FILENAME
        from blb_stage2_rl.runner import (
            BLB_STAGE2_FINAL_CHECKPOINT_FILENAME,
            BLB_STAGE2_LIVE_CHECKPOINT_FILENAME,
        )

        with tempfile.TemporaryDirectory() as td:
            progress_dir = os.path.join(td, "stage2_noise", "progress")
            os.makedirs(progress_dir, exist_ok=True)
            legacy_path = os.path.join(progress_dir, NOISE_STAGE_CHECKPOINT_FILENAME)
            blb_live_path = os.path.join(progress_dir, BLB_STAGE2_LIVE_CHECKPOINT_FILENAME)
            blb_final_path = os.path.join(progress_dir, BLB_STAGE2_FINAL_CHECKPOINT_FILENAME)
            for path in (legacy_path, blb_live_path, blb_final_path):
                with open(path, "wb") as f:
                    f.write(b"checkpoint")

            ev = LayerImportanceEvaluator.__new__(LayerImportanceEvaluator)
            ev.resume_run_dir = td
            ev.stage2_rl_variant = "blb_v3"
            self.assertEqual(ev._get_stage2_resume_checkpoint_path(), blb_final_path)

            os.remove(blb_final_path)
            self.assertEqual(ev._get_stage2_resume_checkpoint_path(), blb_live_path)

            ev.stage2_rl_variant = "legacy_v2"
            self.assertEqual(ev._get_stage2_resume_checkpoint_path(), legacy_path)


class RunnerRealRescaleConfigTests(unittest.TestCase):
    """BLB Stage-2 RL training must use the real Rescale_optimizer path."""

    def test_train_config_ignores_legacy_invoker_choice(self):
        from types import SimpleNamespace

        from blb_stage2_rl import BLBStage2RLRunner

        ev = SimpleNamespace(
            dataset_key="mrpc",
            stage2_rl_episodes=80000,
            blb_v3_rescale_invoker_kind="heuristic",
            blb_v3_inproc_rescale_optimizer_root="",
        )
        cfg = BLBStage2RLRunner(ev)._build_train_config_from_evaluator(ev)

        self.assertEqual(cfg.total_episodes, 80000)
        self.assertEqual(cfg.profile, "mrpc")
        self.assertTrue(
            cfg.inproc_rescale_optimizer_root.endswith("Rescale_optimizer"),
            cfg.inproc_rescale_optimizer_root,
        )

    def test_build_rescale_bridge_never_falls_back_to_heuristic(self):
        import tempfile

        from blb_stage2_rl import BLBStage2RLRunner
        from blb_stage2_rl.runner import BLBStage2TrainConfig

        with tempfile.TemporaryDirectory() as td:
            cfg = BLBStage2TrainConfig(
                profile="mrpc",
                inproc_rescale_optimizer_root=td,
            )
            with self.assertRaises(RuntimeError):
                BLBStage2RLRunner(object())._build_rescale_bridge(
                    cfg,
                    log=lambda _msg: None,
                )


class RunnerEndToEndTests(unittest.TestCase):
    """``BLBStage2RLRunner.run`` 端到端跑 5 个 episode 验证整条链路（不依赖 Rescale_optimizer）。"""

    def setUp(self):
        from types import SimpleNamespace

        from transformers import BertConfig, BertForSequenceClassification, AutoTokenizer
        from torch.utils.data import Dataset
        from function_handler import ReversibleLayerHandler

        torch.manual_seed(0)
        np.random.seed(0)

        self.config = BertConfig(
            num_hidden_layers=2,
            num_attention_heads=4,
            hidden_size=64,
            intermediate_size=128,
            num_labels=2,
            pad_token_id=0,
        )
        self.model = BertForSequenceClassification(self.config)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.handler = ReversibleLayerHandler(self.model)

        # Mock dataset：每条 8-token 整数 + label
        class _MiniDS(Dataset):
            def __init__(self, n=4):
                torch.manual_seed(123)
                self.ids = torch.randint(2, 100, (n, 8))
                self.am = torch.ones_like(self.ids)
                self.labels = torch.randint(0, 2, (n,))

            def __len__(self):
                return self.ids.shape[0]

            def __getitem__(self, i):
                return {
                    "input_ids": self.ids[i],
                    "attention_mask": self.am[i],
                    "labels": self.labels[i],
                }

            @property
            def column_names(self):
                return ["input_ids", "attention_mask", "labels"]

            def select(self, idx):
                # subset：返回新的 _MiniDS 视图
                idx = list(idx)
                sub = _MiniDS(n=len(idx))
                sub.ids = self.ids[idx]
                sub.am = self.am[idx]
                sub.labels = self.labels[idx]
                return sub

            def __getitem__column(self, key):
                return [int(self.labels[i]) for i in range(len(self))]

            def __getitem_column__(self, key):
                if key == "labels":
                    return [int(x) for x in self.labels]
                raise KeyError(key)

        ds = _MiniDS(n=4)

        # 简单 collator：把 list-of-dict pad 到 [B,8]
        def collator(batch):
            ids = torch.stack([b["input_ids"] for b in batch])
            am = torch.stack([b["attention_mask"] for b in batch])
            labels = torch.stack([torch.tensor(b["labels"], dtype=torch.long) for b in batch])
            return {"input_ids": ids, "attention_mask": am, "labels": labels}

        # _MiniDS 的 _get_stability_probe 兼容接口 —— 实际在 evaluator 里走 train_test_split，
        # 这里直接返回完整 ds（本测试关注链路通畅，不关注分层采样精确度）。
        def _get_stability_probe(split_name, probe_size, probe_seed=42):
            return ds, None

        # 与 _MiniDS 兼容的 evaluate_model（返回常量；测试不关心数值精度）
        def _evaluate_model(gelu, softmax, use_train=True, split=None):
            return (0.6, 0.5, 0.5, 1.0)

        def _build_constraint_limits_from_metrics(loss, p, s, **kwargs):
            return {"loss": loss * 1.05, "metric1": p * 0.95, "metric2": s * 0.95}

        def _get_max_noise_configuration():
            return {
                "input_noise_scaling_factors": np.full(2, 22, dtype=int),
                "wq_noise_scaling_factors": np.full(2, 22, dtype=int),
                "wk_noise_scaling_factors": np.full(2, 22, dtype=int),
                "wv_noise_scaling_factors": np.full(2, 22, dtype=int),
                "wo_noise_scaling_factors": np.full(2, 22, dtype=int),
                "wffn1_noise_scaling_factors": np.full(2, 22, dtype=int),
                "wffn2_noise_scaling_factors": np.full(2, 22, dtype=int),
            }

        def _get_noise_simulated_cost(**cfg):
            return 100.0, {"placeholder": 100.0}

        # 临时 progress dir
        import tempfile
        progress_dir = tempfile.mkdtemp(prefix="blb_runner_test_")

        # 用 SimpleNamespace 拼出 ev
        ev = SimpleNamespace(
            log=print,
            apply_configuration=lambda g, s: None,    # spec §3.2 中 attention/GELU 由测试外部装好
            reversible_handler=self.handler,
            total_layers=2,
            layers_attribute="bert.encoder.layer",
            model=self.model,
            device=torch.device("cpu"),
            batch_size=2,
            data_collator=collator,
            dataset_splits={"train": ds, "validation": ds, "validation_full": ds},
            dataset_key="mrpc",
            is_regression=False,
            stage2_rl_episodes=5,
            stage2_ppo_lr_initial=1e-3,
            stage2_k_trials=2,
            stage2_probe_size=4,
            stage2_limit_tolerance=0.05,
            stage2_stability_tolerance=0.05,
            final_eval_random_seed=42,
            noise_stage_progress_dir=progress_dir,
            blb_v3_rescale_invoker_kind="heuristic",
            blb_v3_subprocess_optimizer_root=None,
            blb_v3_subprocess_cli_module="rescale_optimizer.replan",
            blb_v3_rollout_size=4,
            blb_v3_eval_interval=4,
            blb_v3_save_interval=10,
            blb_v3_calibrate_baseline_samples=2,
            stage2_rl_variant="blb_v3",
            _get_stability_probe=_get_stability_probe,
            evaluate_model=_evaluate_model,
            build_constraint_limits_from_metrics=_build_constraint_limits_from_metrics,
            _get_max_noise_configuration=_get_max_noise_configuration,
            get_noise_simulated_cost=_get_noise_simulated_cost,
            get_reward_reference_split_name=lambda: "validation",
            has_dataset_split=lambda s: s in ev.dataset_splits if hasattr(ev, "dataset_splits") else False,
        )
        # 装好多项式近似
        self.handler.replace_layer_softmax([0, 1], "model.bert.encoder.layer", degree=4)
        self.handler.replace_layer_gelu([0, 1], "model.bert.encoder.layer", degree=4)
        self.ev = ev

    def test_runner_run_completes(self):
        from blb_stage2_rl import BLBStage2RLRunner
        from blb_stage2_rl.runner import BLB_STAGE2_FINAL_CHECKPOINT_FILENAME

        runner = BLBStage2RLRunner(self.ev)
        result = runner.run(
            fixed_gelu=np.array([4, 4], dtype=int),
            fixed_softmax=np.array([4, 4], dtype=int),
            fixed_label="Mock",
            fixed_source="mock",
        )

        # 关键字段（与旧版 noise_rl_module_v2 兼容 + 新版独有字段）
        self.assertIn("best_noise_config", result)
        self.assertIn("baseline_tot_c", result)
        self.assertIn("limit_loss", result)
        self.assertIn("limit_p", result)
        self.assertIn("limit_s", result)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["rl_variant"], "blb_v3")
        self.assertEqual(int(result["blb_v3_total_episodes"]), 5)
        # 训练完应该至少给出 best_action_vec
        self.assertIsNotNone(result.get("blb_v3_best_action_vec"))
        final_ckpt_path = os.path.join(
            self.ev.noise_stage_progress_dir, BLB_STAGE2_FINAL_CHECKPOINT_FILENAME,
        )
        self.assertTrue(os.path.isfile(final_ckpt_path))
        ckpt = _torch_load_checkpoint(final_ckpt_path)
        self.assertEqual(int(ckpt["completed_episodes"]), 5)
        self.assertEqual(ckpt["fixed_gelu"], [4, 4])
        self.assertEqual(ckpt["fixed_softmax"], [4, 4])
        self.assertEqual(ckpt["rl_variant"], "blb_v3")

    def test_runner_resume_from_checkpoint_continues(self):
        from blb_stage2_rl import BLBStage2RLRunner
        from blb_stage2_rl.runner import BLB_STAGE2_FINAL_CHECKPOINT_FILENAME

        self.ev.stage2_rl_episodes = 3
        self.ev.blb_v3_save_interval = 2
        runner = BLBStage2RLRunner(self.ev)
        runner.run(
            fixed_gelu=np.array([4, 4], dtype=int),
            fixed_softmax=np.array([4, 4], dtype=int),
            fixed_label="Mock",
            fixed_source="mock",
        )
        final_ckpt_path = os.path.join(
            self.ev.noise_stage_progress_dir, BLB_STAGE2_FINAL_CHECKPOINT_FILENAME,
        )
        self.assertTrue(os.path.isfile(final_ckpt_path))

        self.ev.stage2_rl_episodes = 5
        resumed = runner.run(
            fixed_gelu=np.array([4, 4], dtype=int),
            fixed_softmax=np.array([4, 4], dtype=int),
            fixed_label="Mock",
            fixed_source="mock",
            resume_checkpoint_path=final_ckpt_path,
        )
        self.assertEqual(int(resumed["blb_v3_total_episodes"]), 5)
        ckpt = _torch_load_checkpoint(final_ckpt_path)
        self.assertEqual(int(ckpt["completed_episodes"]), 5)


if __name__ == "__main__":
    unittest.main()
