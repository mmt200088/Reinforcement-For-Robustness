import csv
import contextlib
import io
import json
import os
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


class BLBInstallLogRegressionTests(unittest.TestCase):
    def test_blb_install_log_helper_respects_quiet_env(self):
        import function_handler

        previous = os.environ.get("BLB_NOISE_INSTALL_LOGS")
        try:
            os.environ["BLB_NOISE_INSTALL_LOGS"] = "0"
            quiet = io.StringIO()
            with contextlib.redirect_stdout(quiet):
                function_handler._print_blb_install("hidden")
            self.assertEqual(quiet.getvalue(), "")

            os.environ["BLB_NOISE_INSTALL_LOGS"] = "1"
            loud = io.StringIO()
            with contextlib.redirect_stdout(loud):
                function_handler._print_blb_install("shown")
            self.assertEqual(loud.getvalue(), "shown\n")
        finally:
            if previous is None:
                os.environ.pop("BLB_NOISE_INSTALL_LOGS", None)
            else:
                os.environ["BLB_NOISE_INSTALL_LOGS"] = previous


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

    def test_action_candidate_still_runs_model_forward_when_optimizer_invalid(self):
        from Paean.blb_action_eval import BLBActionFinalEvaluationModule
        from blb_stage2_rl.action_space import make_all_max_action_vector

        class FakeEvaluator:
            total_layers = 1
            dataset_key = "mrpc"

            def get_simulated_cost(self, _gelu, _softmax):
                return 10.0, 4.0, 6.0

        signals = type(
            "Signals",
            (),
            {
                "any_invalid": True,
                "total_bits_sum": 0,
                "total_fusion_count": 0,
                "invalid_block_count": 1,
                "valid_block_count": 0,
            },
        )()

        runner = BLBActionFinalEvaluationModule.__new__(BLBActionFinalEvaluationModule)
        runner.evaluator = FakeEvaluator()
        runner._optimizer_outputs = lambda _profile, _cfgs_dict: ({}, signals)
        runner._config_details = lambda *_args, **_kwargs: {}
        runner._is_feasible = lambda *_args, **_kwargs: True

        runner._run_blb_eval = lambda *_args, **_kwargs: (
            {"loss": 0.25, "p": 0.875, "s": 0.8, "time_ms": 12.0, "install_verification": {"ok": True}},
            None,
        )

        result = runner._evaluate_action_candidate(
            name="candidate",
            action_vec=make_all_max_action_vector(num_layers=1),
            overrides={},
            gelu=np.ones(1, dtype=int),
            softmax=np.ones(1, dtype=int) * 2,
            report_constraints={},
        )

        self.assertTrue(result["any_invalid"])
        self.assertFalse(result["skipped_forward"])
        self.assertFalse(result["feasible"])
        self.assertEqual(result["install_verification"], {"ok": True})
        self.assertEqual(result["p"], 0.875)

    def test_full_noise_config_uses_training_first_input_N(self):
        from Paean.blb_action_eval import BLBActionFinalEvaluationModule
        from blb_stage2_rl.action_space import BLB_FIRST_INPUT_N

        decoded = type(
            "Decoded",
            (),
            {
                "first_input_sf": 30,
                "block1_cfgs": {},
                "block2_cfgs": {},
                "block3_cfgs": {},
                "block4_cfgs": {},
                "block5_cfgs": {},
            },
        )()

        config = BLBActionFinalEvaluationModule._full_noise_config(decoded)

        self.assertEqual(config["entries"][0]["N"], BLB_FIRST_INPUT_N)


class BLBPolicyWarmstartRegressionTests(unittest.TestCase):
    def test_kind_drop_action_changes_only_target_kind_slots(self):
        from blb_stage2_rl.action_space import (
            action_dims_for_config,
            describe_action_vector,
            load_max_sfs,
            make_all_max_action_vector,
        )
        from blb_stage2_rl.runner import _build_kind_drop_action

        baseline = make_all_max_action_vector(num_layers=2)
        desc = describe_action_vector(
            baseline,
            max_sfs=load_max_sfs("mrpc"),
            num_layers=2,
            gelu_degree=[1, 4],
            attn_degree=[2, 5],
            profile="mrpc",
        )
        records = list(desc["records"])
        action, touched = _build_kind_drop_action(
            baseline,
            records,
            action_dims_for_config(2),
            kinds=("M", "S"),
            radius=1,
        )

        self.assertGreater(len(touched), 0)
        self.assertFalse(np.array_equal(action, baseline))
        for idx, (actual, expected) in enumerate(zip(action.tolist(), baseline.tolist())):
            if idx in touched:
                self.assertIn(records[idx]["kind"], ("M", "S"))
                self.assertLess(actual, expected)
            else:
                self.assertEqual(actual, expected)

    def test_warmstart_action_mode_expires_by_absolute_episode(self):
        from blb_stage2_rl.runner import _warmstart_action_mode

        kwargs = {
            "anchor_episodes": 30,
            "cost_probe_count": 4,
            "neighbor_sampling": True,
            "has_mutable_neighbors": True,
            "neighbor_ramp_episodes": 1200,
        }

        self.assertEqual(
            _warmstart_action_mode(episode_index=0, **kwargs),
            ("anchor", -1),
        )
        self.assertEqual(
            _warmstart_action_mode(episode_index=30, **kwargs),
            ("cost_probe", 0),
        )
        self.assertEqual(
            _warmstart_action_mode(episode_index=34, **kwargs),
            ("neighbor", -1),
        )
        self.assertEqual(
            _warmstart_action_mode(episode_index=1199, **kwargs),
            ("neighbor", -1),
        )
        self.assertEqual(
            _warmstart_action_mode(episode_index=1200, **kwargs),
            ("policy", -1),
        )
        self.assertEqual(
            _warmstart_action_mode(episode_index=50000, **kwargs),
            ("policy", -1),
        )

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
    def test_stage2_noise_log_formatters_use_readable_chinese(self):
        from blb_stage2_rl.runner import (
            _format_blb_best_log,
            _format_blb_episode_error_log,
            _format_blb_rollout_summary_log,
            _format_blb_train_iter_log,
            _format_warmstart_cost_probe_log,
        )

        error_text = (
            "Rescale_optimizer invalid blocks: "
            "block5_n1_L7: new chain has prime(s) > q_max=60 at stage(s) [1]; "
            "fusion cannot reduce. Reject.; "
            "block3_exp_n2_L0: replan FAILED: a prime < q_min=30 could not be fused "
            "after 0 successful fusion(s)."
        )
        formatted_error = _format_blb_episode_error_log(485, error_text)

        self.assertIn("【BLB 单回合错误】", formatted_error)
        self.assertIn("回合（episode）：485", formatted_error)
        self.assertIn("失败位置", formatted_error)
        self.assertIn("block5_n1_L7", formatted_error)
        self.assertIn("block3_exp_n2_L0", formatted_error)
        self.assertIn("新模数链", formatted_error)
        self.assertIn("重新规划（replan）失败", formatted_error)
        self.assertNotIn("[BLB episode error]", formatted_error)
        self.assertNotIn("new chain has prime(s)", formatted_error)

        formatted_rollout = _format_blb_rollout_summary_log(
            update_count=5,
            episode=600,
            total_episodes=80000,
            reward_mean=-36.691,
            reward_max=-3.400,
            reward_min=-100.188,
            invalid_count=11,
            priority_counts={1: 1, 2: 77, 3: 31},
            anchor_count=0,
            cost_probe_count=0,
            neighborhood_count=120,
            policy_loss=0.0128,
            value_loss=1711.9232,
            entropy=1071.4326,
            clip_fraction=0.25,
            entropy_by_kind={"F": 1.44, "W": 1.48, "M": 0.92},
        )

        self.assertIn("【BLB Rollout 汇总】", formatted_rollout)
        self.assertIn("训练进度（episode）：600 / 80000", formatted_rollout)
        self.assertIn("奖励（reward，均值 / 最大 / 最小）：-36.691 / -3.400 / -100.188", formatted_rollout)
        self.assertIn("优先级计数（P0/P1/P2/P3）", formatted_rollout)
        self.assertIn("P0 无效=11", formatted_rollout)
        self.assertIn("动作来源（A/C/N）", formatted_rollout)
        self.assertIn("槽位熵（H_kind）", formatted_rollout)
        self.assertIn("F=1.44", formatted_rollout)

        formatted_probes = _format_warmstart_cost_probe_log(
            [("drop_kind_M", object(), [1, 2]), ("drop_kind_MS", object(), [3])]
        )
        self.assertIn("预热（warmstart）成本探针", formatted_probes)
        self.assertIn("drop_kind_M：影响槽位 2 个", formatted_probes)

        formatted_best = _format_blb_best_log(
            episode=31,
            best_reward=2.7333,
            previous_reward_label="0.0000",
            priority=3,
            diff_text="L0.B2.M.gamma 2->1; L0.B2.M.q_mask1 2->1",
        )
        self.assertIn("【BLB 新最佳】", formatted_best)
        self.assertIn("回合（episode）：31", formatted_best)
        self.assertIn("当前奖励（reward）：2.7333", formatted_best)
        self.assertIn("上一最佳奖励：0.0000", formatted_best)
        self.assertIn("变化位置", formatted_best)
        self.assertIn("L0.B2.M.gamma", formatted_best)
        self.assertNotIn("[BLB best]", formatted_best)

        formatted_iter = _format_blb_train_iter_log(
            episode=120,
            total_episodes=80000,
            return_mean=-25.817,
            return_max=2.733,
            best_reward=2.733,
            policy_loss=0.2411,
            value_loss=1298.1715,
            entropy=1071.0776,
            clip_fraction=0.8708,
        )
        self.assertIn("【BLB 训练迭代】", formatted_iter)
        self.assertIn("近期回报（return，均值 / 最大）", formatted_iter)
        self.assertIn("best_reward=2.733", formatted_iter)

        evaluator_source = (Path(__file__).resolve().parents[1] / "layer_importance_evaluator.py").read_text(
            encoding="utf-8",
        )
        runner_source = (Path(__file__).resolve().parents[1] / "blb_stage2_rl" / "runner.py").read_text(
            encoding="utf-8",
        )
        self.assertIn("二阶段噪声 RL 日志开始（Stage-2 noise RL log started）", evaluator_source)
        self.assertIn("一阶段 PPO 学习率（Stage-1 PPO LR）", evaluator_source)
        self.assertNotIn("?????? RL", evaluator_source)
        self.assertIn("BLB 单候选安装日志", runner_source)
        self.assertNotIn("per-candidate install logs suppressed", runner_source)

    def test_status_board_publishes_live_top_level_fields(self):
        from blb_stage2_rl.persistence import BLBStatusBoard

        with _workspace_tempdir() as td:
            board = BLBStatusBoard(td, total_episodes=240, profile="mrpc")
            board.set_best(
                best_reward=0.0,
                best_action_vec=[1, 2, 3],
                best_breakdown={"priority": 3, "invalid": False},
                best_episode=0,
            )
            board.update_after_episode(
                120,
                -30.0,
                breakdown={"priority": 0, "invalid": True},
            )
            board.update_after_ppo_update(1, {"policy_loss": 0.25})

            payload = json.loads(Path(board.path).read_text(encoding="utf-8"))

        self.assertEqual(payload["episode"], 120)
        self.assertEqual(payload["completed_episodes"], 120)
        self.assertEqual(payload["best_reward"], 0.0)
        self.assertEqual(payload["best"]["reward"], 0.0)
        self.assertEqual(payload["last_reward"], -30.0)
        self.assertEqual(payload["last_priority"], 0)
        self.assertTrue(payload["last_invalid"])
        self.assertIsNotNone(payload["updated_at"])

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

    def test_trace_writer_migrates_old_header_after_cost_probe_column_added(self):
        from blb_stage2_rl.persistence import (
            BLB_EPISODE_TRACE_CSV,
            BLB_TRACE_FIELDNAMES,
            append_blb_episode_trace_row,
        )

        with _workspace_tempdir() as td:
            trace_path = Path(td) / BLB_EPISODE_TRACE_CSV
            old_header = [field for field in BLB_TRACE_FIELDNAMES if field != "cost_probe_count"]
            old_row = {field: "" for field in old_header}
            old_row.update({
                "episode": 120,
                "total_episodes": 80000,
                "ppo_update_count": 1,
                "anchor_count": 30,
                "policy_loss": 0.01,
                "value_loss": 10.0,
                "entropy": 100.0,
                "clip_fraction": 0.1,
                "n_samples": 120,
            })
            malformed_new_row = {field: "" for field in BLB_TRACE_FIELDNAMES}
            malformed_new_row.update({
                "episode": 520,
                "total_episodes": 80000,
                "ppo_update_count": 4,
                "anchor_count": 0,
                "cost_probe_count": 4,
                "policy_loss": 0.02,
                "value_loss": 2437.0,
                "entropy": 1071.0,
                "clip_fraction": 0.744,
                "n_samples": 120,
            })
            with trace_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(old_header)
                writer.writerow([old_row.get(field, "") for field in old_header])
                writer.writerow([malformed_new_row.get(field, "") for field in BLB_TRACE_FIELDNAMES])

            append_blb_episode_trace_row(
                td,
                {
                    "episode": 640,
                    "total_episodes": 80000,
                    "ppo_update_count": 5,
                    "anchor_count": 0,
                    "cost_probe_count": 0,
                    "policy_loss": 0.03,
                    "value_loss": 20.0,
                    "entropy": 90.0,
                    "clip_fraction": 0.2,
                    "n_samples": 120,
                },
            )

            with trace_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(list(rows[0].keys()), list(BLB_TRACE_FIELDNAMES))
        self.assertEqual(rows[0]["cost_probe_count"], "0")
        self.assertEqual(rows[0]["policy_loss"], "0.01")
        self.assertEqual(rows[1]["cost_probe_count"], "4")
        self.assertEqual(rows[1]["policy_loss"], "0.02")
        self.assertEqual(rows[1]["value_loss"], "2437.0")
        self.assertEqual(rows[1]["entropy"], "1071.0")
        self.assertEqual(rows[1]["clip_fraction"], "0.744")
        self.assertEqual(rows[1]["n_samples"], "120")
        self.assertEqual(rows[2]["episode"], "640")

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
    def test_optimizer_invalid_is_cost_layer_after_accuracy_and_stability(self):
        from blb_stage2_rl.reward import (
            BaselineCostStats,
            EpisodeMetrics,
            RewardWeights,
            compute_reward,
        )

        accuracy_breakdown = compute_reward(
            EpisodeMetrics(metric1_mean=0.0, loss_mean=float("inf"), loss_std=float("inf")),
            type("Signals", (), {"any_invalid": True, "total_bits_sum": 0, "total_fusion_count": 0})(),
            action_avg_k=13.0,
            baseline=BaselineCostStats(metric1_mean=0.875),
            weights=RewardWeights(invalid_penalty=30.0),
            acc_threshold=0.865,
            stab_threshold=1.0,
            any_invalid=True,
        )

        self.assertTrue(accuracy_breakdown.invalid)
        self.assertEqual(accuracy_breakdown.priority, 1)
        self.assertGreater(accuracy_breakdown.acc_violation, 0.0)

        cost_breakdown = compute_reward(
            EpisodeMetrics(metric1_mean=0.9, loss_mean=0.2, loss_std=0.1),
            type("Signals", (), {"any_invalid": True, "total_bits_sum": 200, "total_fusion_count": 0})(),
            action_avg_k=13.0,
            baseline=BaselineCostStats(total_bits_sum=200, metric1_mean=0.875),
            weights=RewardWeights(invalid_penalty=30.0),
            acc_threshold=0.865,
            stab_threshold=1.0,
            any_invalid=True,
        )

        self.assertTrue(cost_breakdown.invalid)
        self.assertEqual(cost_breakdown.priority, 3)
        self.assertEqual(cost_breakdown.r_invalid, -30.0)

        nonfinite_stability = compute_reward(
            EpisodeMetrics(metric1_mean=0.9, loss_mean=0.2, loss_std=float("inf")),
            type("Signals", (), {"any_invalid": False, "total_bits_sum": 200, "total_fusion_count": 0})(),
            action_avg_k=13.0,
            baseline=BaselineCostStats(total_bits_sum=200, metric1_mean=0.875),
            weights=RewardWeights(priority2_penalty=50.0, priority2_scale=100.0),
            acc_threshold=0.865,
            stab_threshold=1.0,
            any_invalid=False,
        )
        self.assertEqual(nonfinite_stability.priority, 2)
        self.assertEqual(nonfinite_stability.reward, -150.0)

    def test_runner_best_selection_uses_hard_constraints_before_reward(self):
        from blb_stage2_rl.runner import is_better_blb_candidate

        invalid_high_reward = {
            "invalid": True,
            "acc_violation": 0.0,
            "stab_violation": 0.0,
        }
        valid_accuracy_failure = {
            "invalid": False,
            "acc_violation": 0.2,
            "stab_violation": 0.0,
        }
        valid_safe_lower_reward = {
            "invalid": False,
            "acc_violation": 0.0,
            "stab_violation": 0.0,
        }
        valid_safe_higher_reward = {
            "invalid": False,
            "acc_violation": 0.0,
            "stab_violation": 0.0,
        }

        self.assertFalse(
            is_better_blb_candidate(
                candidate_reward=-100.0,
                candidate_breakdown=valid_accuracy_failure,
                best_reward=-30.0,
                best_breakdown=invalid_high_reward,
            )
        )
        self.assertTrue(
            is_better_blb_candidate(
                candidate_reward=1.0,
                candidate_breakdown=valid_safe_higher_reward,
                best_reward=0.5,
                best_breakdown=valid_safe_lower_reward,
            )
        )
        self.assertTrue(
            is_better_blb_candidate(
                candidate_reward=-30.0,
                candidate_breakdown=invalid_high_reward,
                best_reward=-100.0,
                best_breakdown=valid_accuracy_failure,
            )
        )

    def test_baseline_preflight_stability_threshold_uses_noisy_baseline(self):
        from blb_stage2_rl.runner import _baseline_preflight_stability_threshold

        widened = _baseline_preflight_stability_threshold(
            current_threshold=0.001,
            observed_loss_std=0.0018,
            tolerance=0.005,
        )

        self.assertGreater(widened, 0.0018)
        self.assertAlmostEqual(
            _baseline_preflight_stability_threshold(
                current_threshold=0.003,
                observed_loss_std=0.0018,
                tolerance=0.005,
            ),
            0.003,
        )

    def test_neighbor_indices_use_real_k_order_not_action_index_order(self):
        from blb_stage2_rl.action_space import K_LEVELS
        from blb_stage2_rl.runner import _allowed_neighbor_indices

        baseline_idx = int(K_LEVELS.index(max(K_LEVELS)))
        allowed = _allowed_neighbor_indices(
            kind="K",
            baseline_idx=baseline_idx,
            dim=len(K_LEVELS),
            radius=1,
        )

        self.assertEqual([K_LEVELS[i] for i in allowed], [13, 12])

    def test_neighborhood_curriculum_expands_slowly(self):
        from blb_stage2_rl.runner import _neighborhood_curriculum

        self.assertEqual(
            _neighborhood_curriculum(
                episode_offset=0,
                ramp_episodes=100,
                max_mutations=8,
                max_radius=2,
            ),
            (1, 1),
        )
        self.assertEqual(
            _neighborhood_curriculum(
                episode_offset=100,
                ramp_episodes=100,
                max_mutations=8,
                max_radius=2,
            ),
            (8, 2),
        )


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
                        total_bits=100,
                        invalid_chain=None,
                        valid=True,
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
            # 4 blocks (block2..5) * 100 bits each — layer-0 block1 is no longer
            # installed (the first HE config is treated as lossless), so the
            # all-max baseline reports 4 valid blocks, not 5.
            baseline=BaselineCostStats(total_bits_sum=400, total_fusion_count=0, avg_k=13.0),
            reward_weights=RewardWeights(),
            acc_threshold=0.5,
            stab_threshold=10.0,
            max_sfs=load_max_sfs("mrpc"),
            num_layers=1,
            env_cfg=BLBStage2EnvConfig(profile="mrpc", num_trials_per_step=1),
        )

        _obs, reward, done, info = env.step(make_all_max_action_vector(num_layers=1))

        self.assertEqual(bridge.calls, ["cfg-derived"])
        self.assertTrue(done)
        self.assertFalse(info["invalid"])
        self.assertFalse(info["reward_breakdown"].invalid)
        self.assertAlmostEqual(reward, 0.0)

    def test_env_runs_forward_even_when_optimizer_invalid(self):
        from blb_stage2_rl.action_space import load_max_sfs, make_all_max_action_vector
        from blb_stage2_rl.env import BLBStage2Env, BLBStage2EnvConfig, ProbeBatch
        from blb_stage2_rl.reward import BaselineCostStats, RewardWeights
        from rescale_optimizer_bridge import RescaleOptimizerOutput

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(()))
                self.forward_count = 0

            def forward(self, **_kwargs):
                self.forward_count += 1
                logits = torch.tensor([[0.0, 10.0], [10.0, 0.0]])
                return type("Output", (), {"logits": logits})()

        class FakeHandler:
            def __getattr__(self, name):
                if name.startswith(("replace_", "restore_")):
                    return lambda *args, **kwargs: None
                raise AttributeError(name)

        class FakeRescaleBridge:
            def evaluate_blocks(self, requests):
                return {
                    name: RescaleOptimizerOutput(
                        config_name=name,
                        fusion_count=0,
                        total_bits=100,
                        invalid_chain={"reason": "unit"},
                        valid=False,
                        raw={},
                    )
                    for name in requests
                }

        probe = ProbeBatch(
            input_ids=torch.ones((2, 4), dtype=torch.long),
            attention_mask=torch.ones((2, 4), dtype=torch.long),
            labels=torch.tensor([1, 0], dtype=torch.long),
        )
        model = TinyModel()
        env = BLBStage2Env(
            handler=FakeHandler(),
            model=model,
            probe_batches=[probe],
            rescale_bridge=FakeRescaleBridge(),
            baseline=BaselineCostStats(total_bits_sum=400, total_fusion_count=0, avg_k=13.0),
            reward_weights=RewardWeights(invalid_penalty=30.0),
            acc_threshold=0.5,
            stab_threshold=10.0,
            max_sfs=load_max_sfs("mrpc"),
            num_layers=1,
            env_cfg=BLBStage2EnvConfig(profile="mrpc", num_trials_per_step=1),
        )

        _obs, _reward, done, info = env.step(make_all_max_action_vector(num_layers=1))

        self.assertTrue(done)
        self.assertTrue(info["invalid"])
        self.assertTrue(info["forward_ran"])
        self.assertEqual(model.forward_count, 1)
        self.assertEqual(info["reward_breakdown"].priority, 3)
        self.assertTrue(info["reward_breakdown"].invalid)
        self.assertEqual(info["reward_breakdown"].r_invalid, -30.0)

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
        # 5 blocks * 12 layers - 1 = 59 (layer-0 block 1 is no longer installed:
        # the first HE config is treated as lossless, aligned with Rescale_optimizer).
        self.assertEqual(signals.valid_block_count, 59)
        self.assertEqual(signals.invalid_block_count, 0)

    def test_real_mrpc_all_max_cfg_derived_optimizer_outputs_are_valid(self):
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

        decoded = action_vector_to_cfgs(
            make_all_max_action_vector(num_layers=12),
            load_max_sfs("mrpc"),
            num_layers=12,
            gelu_degree=4,
            attn_degree=4,
        )
        requests = build_optimizer_requests("mrpc", decoded.cfgs_dict())
        bridge = RescaleOptimizerBridge(
            invoker=InProcessInvoker.from_profile(
                rescale_optimizer_root="Rescale_optimizer",
                profile="mrpc",
            )
        )

        outputs = bridge.evaluate_blocks(requests)
        signals = aggregate_optimizer_signals(outputs)

        self.assertFalse(signals.any_invalid, signals.invalid_chains)
        # 5 blocks * 12 layers - 1 = 59 (layer-0 block 1 is no longer installed:
        # the first HE config is treated as lossless, aligned with Rescale_optimizer).
        self.assertEqual(signals.valid_block_count, 59)
        self.assertEqual(signals.invalid_block_count, 0)
        self.assertEqual(
            outputs["block4_L0"].raw["delta_overrides"]["ctct_rot_softmax_mul_v"],
            39,
        )


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


class BLBPlaybookArtifactRegressionTests(unittest.TestCase):
    def test_phase0_preflight_report_names_current_operator_entrypoints(self):
        from scripts.blb_phase0_preflight import build_phase0_entrypoint_report

        repo_root = Path(__file__).resolve().parents[1]
        report = build_phase0_entrypoint_report(repo_root)

        self.assertIn("llama_7B_LayerImportance.sh", report)
        self.assertIn("run rl --preset mrpc-blb-stage2-rl", report)
        self.assertIn("blb_stage2_rl/runner.py", report)
        self.assertIn("Rescale_optimizer", report)

    def test_candidate_store_hash_fidelity_and_rank_key_are_stable(self):
        from blb_stage2_rl.candidate_store import (
            CandidateStore,
            action_hash,
            candidate_rank_key,
        )

        action = [4, 3, 2, 1]
        self.assertEqual(action_hash(action), action_hash(np.array(action, dtype=int)))
        self.assertLess(
            candidate_rank_key({"valid": True, "acc_violation": 0.0, "stability_violation": 0.0, "normalized_cost": 0.4}),
            candidate_rank_key({"valid": True, "acc_violation": 0.0, "stability_violation": 0.0, "normalized_cost": 0.8}),
        )
        self.assertGreater(
            candidate_rank_key({"valid": True, "acc_violation": 0.0, "stability_violation": 0.1, "normalized_cost": 0.1}),
            candidate_rank_key({"valid": False, "acc_violation": 0.0, "stability_violation": 0.0, "normalized_cost": 0.0}),
        )

        with _workspace_tempdir() as td:
            store = CandidateStore(Path(td) / "candidate_store.jsonl")
            self.assertTrue(store.should_evaluate(action, "F1"))
            store.append({"action_indices": action, "fidelity": "F1", "valid": True})
            self.assertFalse(store.should_evaluate(action, "F1"))
            self.assertTrue(store.should_evaluate(action, "F2"))
            store.append({"action_indices": action, "fidelity": "F2", "valid": True})
            self.assertFalse(store.should_evaluate(action, "F2"))

    def test_registry_export_records_action_values_and_current_slot_count(self):
        from blb_stage2_rl.action_space import K_LEVELS
        from scripts.blb_export_action_registry import build_registry_payload

        payload = build_registry_payload(
            profile="mrpc",
            num_layers=1,
            gelu_degree=[4],
            attn_degree=[2],
        )
        records = payload["slot_registry_full"]

        self.assertEqual(payload["schema"], "blb_action_registry_export_v1")
        self.assertTrue(records)
        self.assertTrue(all(r["action_values"] for r in records))
        self.assertTrue(all("scale_semantics" in r for r in records))

        k_records = [r for r in records if r["value_type"] == "truncation_k"]
        self.assertTrue(k_records)
        expected_k_idx = list(K_LEVELS).index(max(K_LEVELS))
        self.assertTrue(all(r["all_max_action_index"] == expected_k_idx for r in k_records))

        self.assertEqual(payload["summary"]["per_layer_slot_count"], 73)
        self.assertEqual(payload["summary"]["block_slot_counts_per_layer"]["block1"], 9)
        self.assertEqual(payload["summary"]["block_slot_counts_per_layer"]["block2"], 23)
        self.assertEqual(payload["summary"]["block_slot_counts_per_layer"]["block3"], 8)
        self.assertEqual(payload["summary"]["block_slot_counts_per_layer"]["block4"], 17)
        self.assertEqual(payload["summary"]["block_slot_counts_per_layer"]["block5"], 16)
        slot_check = payload["current_code_slot_check_markdown"]
        self.assertIn("expected_slots_per_layer", slot_check)
        self.assertIn("status", slot_check)

    def test_f0_eval_record_contains_action_hash_rank_key_and_optimizer_summary(self):
        from blb_stage2_rl.candidate_store import action_hash
        from scripts.blb_eval_action import build_f0_candidate_record

        action = [4, 4, 3, 2]
        signals = type(
            "Signals",
            (),
            {
                "any_invalid": False,
                "total_bits_sum": 1200,
                "total_fusion_count": 7,
                "invalid_chains": {},
            },
        )()

        record = build_f0_candidate_record(
            action,
            source="unit",
            signals=signals,
            baseline_total_bits=2400,
        )

        self.assertEqual(record["fidelity"], "F0")
        self.assertEqual(record["action_hash"], action_hash(action))
        self.assertEqual(record["action_vector_hash"], action_hash(action))
        self.assertEqual(record["normalized_cost"], 0.5)
        self.assertEqual(record["optimizer"]["total_bits_sum"], 1200)
        self.assertEqual(record["rank_key"], [0.0, 1200.0, 7.0])

    def test_f0_eval_all_max_uses_optimizer_baseline_path(self):
        import scripts.blb_eval_action as mod
        from rescale_optimizer_bridge import RescaleOptimizerOutput

        calls = []

        class FakeInvoker:
            @classmethod
            def from_profile(cls, **_kwargs):
                return object()

        class FakeBridge:
            def __init__(self, invoker):
                self.invoker = invoker

            def evaluate_blocks(self, requests):
                calls.append("cfg-derived")
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

            def evaluate_baseline_blocks(self, requests):
                calls.append("baseline")
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

        old_invoker = mod.InProcessInvoker
        old_bridge = mod.RescaleOptimizerBridge
        mod.InProcessInvoker = FakeInvoker
        mod.RescaleOptimizerBridge = FakeBridge
        try:
            with _workspace_tempdir() as td:
                record = mod.run_f0_eval(
                    profile="mrpc",
                    num_layers=1,
                    action_json=None,
                    output_dir=td,
                    source="all_max_f0",
                    rescale_optimizer_root="unused",
                    baseline_total_bits=None,
                )
        finally:
            mod.InProcessInvoker = old_invoker
            mod.RescaleOptimizerBridge = old_bridge

        self.assertEqual(calls, ["cfg-derived"])
        self.assertTrue(record["valid"])


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
