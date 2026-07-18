from __future__ import annotations

import ast
import importlib.machinery
import importlib.util
import math
import pathlib
import sys
import unittest
from dataclasses import dataclass

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _class_literal_defaults(rel_path: str, class_name: str) -> dict:
    tree = ast.parse((REPO_ROOT / rel_path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            out = {}
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    try:
                        out[stmt.target.id] = ast.literal_eval(stmt.value)
                    except Exception:
                        pass
            return out
    raise AssertionError(f"{class_name} not found in {rel_path}")


def _function_literal_defaults(rel_path: str, function_name: str) -> dict:
    tree = ast.parse((REPO_ROOT / rel_path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            args = list(node.args.args)
            defaults = list(node.args.defaults)
            if len(defaults) > len(args):
                raise AssertionError(f"unexpected default layout for {function_name}")
            out = {}
            for arg, default in zip(args[-len(defaults):], defaults):
                try:
                    out[arg.arg] = ast.literal_eval(default)
                except Exception:
                    pass
            return out
    raise AssertionError(f"{function_name} not found in {rel_path}")


def _class_method_literal_defaults(rel_path: str, class_name: str, method_name: str) -> dict:
    tree = ast.parse((REPO_ROOT / rel_path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for stmt in node.body:
                if isinstance(stmt, ast.FunctionDef) and stmt.name == method_name:
                    args = list(stmt.args.args)
                    defaults = list(stmt.args.defaults)
                    out = {}
                    for arg, default in zip(args[-len(defaults):], defaults):
                        try:
                            out[arg.arg] = ast.literal_eval(default)
                        except Exception:
                            pass
                    return out
    raise AssertionError(f"{class_name}.{method_name} not found in {rel_path}")


def _load_module_standalone(rel_path: str, name: str):
    loader = importlib.machinery.SourceFileLoader(name, str(REPO_ROOT / rel_path))
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod


def _active_config_lines(rel_path: str) -> list[str]:
    text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _stage1_exact_barrier(curr_value: float, limit_value: float, *, upper: bool) -> float:
    margin = (limit_value - curr_value) if upper else (curr_value - limit_value)
    if margin < 0.0:
        return -10.0 * math.exp(-margin * 20.0)
    return 0.5 * math.log(margin + 1e-5)


def _load_runner_threshold_helper():
    tree = ast.parse((REPO_ROOT / "blb_stage2_rl/runner.py").read_text(encoding="utf-8"))
    wanted = {
        "BaselineMetricThreshold",
        "_selection_float",
        "_baseline_derived_metric_threshold",
    }
    body = [
        node for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted
    ]
    found = {node.name for node in body}
    if found != wanted:
        raise AssertionError(f"missing runner helper nodes: {sorted(wanted - found)}")
    mod = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {"dataclass": dataclass, "Any": object, "math": math}
    exec(compile(mod, "<runner_threshold_helper>", "exec"), ns)
    return ns["_baseline_derived_metric_threshold"]


class Stage2Stage1AlignmentDefaultsTest(unittest.TestCase):
    def test_stage2_train_config_defaults_disable_search_scaffolds(self):
        defaults = _class_literal_defaults("blb_stage2_rl/runner.py", "BLBStage2TrainConfig")

        self.assertIs(defaults["warmstart_baseline_bias"], False)
        self.assertEqual(defaults["warmstart_anchor_episodes"], 0)
        self.assertIs(defaults["warmstart_neighbor_sampling"], False)
        self.assertIs(defaults["static_invalid_level_mask_enabled"], False)
        self.assertIs(defaults["fusion_neighbor_curriculum_enabled"], False)
        self.assertEqual(defaults["fusion_probe_interval"], 0)
        self.assertEqual(defaults["fusion_exploration_epsilon"], 0.0)
        self.assertEqual(defaults["fusion_exploration_epsilon_k"], 0.0)
        self.assertEqual(defaults["reward_design"], "robust_constrained")
        self.assertEqual(defaults["num_trials_per_step"], 5)
        self.assertEqual(defaults["online_num_trials_per_step"], 5)

    def test_rl_tune_entrypoint_defaults_do_not_reenable_stage2_scaffolds(self):
        defaults = _function_literal_defaults("rl_tune.py", "train")

        self.assertEqual(defaults["blb_v3_sequential_cost_shaping_coeff"], 0.0)
        self.assertIs(defaults["blb_v3_warmstart_baseline_bias"], None)
        self.assertIs(defaults["blb_v3_fusion_neighbor_curriculum"], False)
        self.assertEqual(defaults["blb_v3_fusion_probe_interval"], 0)
        self.assertEqual(defaults["blb_v3_fusion_exploration_epsilon"], 0.0)
        self.assertEqual(defaults["blb_v3_online_k_trials"], 5)

    def test_evaluator_constructor_defaults_do_not_reenable_stage2_scaffolds(self):
        defaults = _class_method_literal_defaults(
            "layer_importance_evaluator.py",
            "LayerImportanceEvaluator",
            "__init__",
        )

        self.assertIs(defaults["blb_v3_fusion_neighbor_curriculum"], False)
        self.assertEqual(defaults["blb_v3_fusion_probe_interval"], 0)
        self.assertEqual(defaults["blb_v3_fusion_exploration_epsilon"], 0.0)

    def test_warmstart_baseline_bias_cli_is_threaded_to_runner(self):
        rl_tune = (REPO_ROOT / "rl_tune.py").read_text(encoding="utf-8")
        evaluator = (REPO_ROOT / "layer_importance_evaluator.py").read_text(encoding="utf-8")
        runner = (REPO_ROOT / "blb_stage2_rl" / "runner.py").read_text(encoding="utf-8")

        self.assertIn("blb_v3_warmstart_baseline_bias: bool = None", rl_tune)
        self.assertIn(
            "blb_v3_warmstart_baseline_bias=blb_v3_warmstart_baseline_bias",
            rl_tune,
        )
        self.assertIn("blb_v3_warmstart_baseline_bias=None", evaluator)
        self.assertIn("self.blb_v3_warmstart_baseline_bias", evaluator)
        self.assertIn("getattr(ev, \"blb_v3_warmstart_baseline_bias\", None)", runner)

    def test_sequential_train_config_defaults_disable_search_scaffolds(self):
        defaults = _class_literal_defaults(
            "blb_stage2_rl/sequential_runner.py",
            "SequentialTrainConfig",
        )

        self.assertIs(defaults["warmstart_neighbor_sampling"], False)
        self.assertIs(defaults["fusion_neighbor_curriculum_enabled"], False)
        self.assertEqual(defaults["fusion_probe_interval"], 0)
        self.assertEqual(defaults["fusion_exploration_epsilon"], 0.0)
        self.assertEqual(defaults["fusion_exploration_epsilon_k"], 0.0)
        self.assertIs(defaults["static_invalid_level_mask_enabled"], False)
        self.assertIs(defaults["empirical_invalid_level_mask_enabled"], False)
        self.assertEqual(defaults["reward_design"], "stage1_aligned")
        self.assertEqual(defaults["online_num_trials_per_step"], 5)

    def test_reward_weights_default_to_stage1_aligned(self):
        reward_mod = _load_module_standalone(
            "blb_stage2_rl/reward.py",
            "stage2_reward_alignment_defaults",
        )

        self.assertEqual(reward_mod.DEFAULT_REWARD_DESIGN, "stage1_aligned")
        self.assertEqual(reward_mod.RewardWeights().reward_design, "stage1_aligned")

    def test_stage1_aligned_parallel_path_respects_configured_baseline_anchor(self):
        text = (REPO_ROOT / "blb_stage2_rl/parallel_runner.py").read_text(encoding="utf-8")

        self.assertNotIn("0.0 if stage1_aligned else _resolve_baseline_prior_scale", text)
        self.assertNotIn("not stage1_aligned\n        and int(force_baseline_episodes)", text)
        self.assertIn("baseline_prior_scale = _resolve_baseline_prior_scale", text)

    def test_stage1_aligned_sequential_path_has_no_hidden_baseline_prior(self):
        text = (REPO_ROOT / "blb_stage2_rl" / "sequential_runner.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("_baseline_prior_enabled = (", text)
        self.assertIn("bool(getattr(train_cfg, \"warmstart_baseline_bias\", False))", text)
        self.assertIn("or int(force_baseline_episodes) > 0", text)
        self.assertIn("if _baseline_prior_enabled else 0.0", text)

    def test_fusion_warmstart_prior_covers_option_and_k_then_decays(self):
        text = (REPO_ROOT / "blb_stage2_rl" / "sequential_runner.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("preferred = [0, int(_baseline_k_index_for_block(1))]", text)
        self.assertNotIn("preferred = [-1, int(_baseline_k_index_for_block(1))]", text)
        self.assertIn("return 0.0", text)

    def test_stage2_training_disables_borderline_retest_for_fixed_k_trials(self):
        text = (REPO_ROOT / "blb_stage2_rl" / "sequential_runner.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("borderline_retest_enabled=False", text)
        self.assertIn("borderline_retest_trials_multiplier=1", text)
        self.assertIn('"borderline_retest_enabled"', text)

    def test_stage1_aligned_runner_disables_fusion_scaffolds(self):
        text = (REPO_ROOT / "blb_stage2_rl" / "sequential_runner.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("_fc_curriculum_on = False if _continuous else", text)
        self.assertIn("fusion_probe_interval=(0 if _continuous else", text)
        self.assertIn("fusion_exploration_epsilon=(0.0 if _continuous else", text)

    def test_default_stage2_preset_does_not_reenable_stage2_scaffolds(self):
        active = _active_config_lines("presets/mrpc-blb-stage2-rl.conf")

        self.assertIn("--blb-v3-sequential-cost-shaping-coeff 0.0", active)
        for forbidden in (
            "--blb-v3-warmstart-anchor-episodes",
            "--blb-v3-fusion-neighbor-curriculum",
            "--blb-v3-fusion-probe-interval",
            "--blb-v3-fusion-exploration-epsilon",
        ):
            self.assertFalse(
                any(line.startswith(forbidden) for line in active),
                f"default preset re-enables Stage-2 scaffold: {forbidden}",
            )

    def test_default_stage2_preset_pins_verified_probe_batch_sizes(self):
        active = _active_config_lines("presets/mrpc-blb-stage2-rl.conf")

        self.assertEqual(
            [
                line
                for line in active
                if line.startswith("--blb-v3-probe-batch-size")
                or line.startswith("--blb-v3-validation-probe-batch-size")
            ],
            [
                "--blb-v3-probe-batch-size 64",
                "--blb-v3-validation-probe-batch-size 64",
            ],
        )

    def test_server_60k_command_does_not_reenable_stage2_scaffolds(self):
        text = (REPO_ROOT / "SERVER_COMMAND.md").read_text(encoding="utf-8")
        marker = "[phase60k]"
        start = text.find(marker)
        self.assertNotEqual(start, -1, "SERVER_COMMAND.md is missing phase60k block")
        block = text[start:start + 2500]

        for forbidden in (
            "--blb-v3-warmstart-anchor-episodes",
            "--blb-v3-fusion-neighbor-curriculum",
            "--blb-v3-fusion-probe-interval",
            "--blb-v3-fusion-exploration-epsilon",
        ):
            self.assertNotIn(
                forbidden,
                block,
                f"phase60k command re-enables Stage-2 scaffold: {forbidden}",
            )

    def test_legacy_runner_preflight_metric_gate_uses_relative_tolerance(self):
        helper = _load_runner_threshold_helper()

        resolved = helper(
            current_threshold=float("nan"),
            raw_baseline_metric=0.90,
            all_max_blb_metric=0.80,
            allowed_drop=0.001,
        )

        self.assertEqual(resolved.source, "baseline_derived_all_max_blb")
        self.assertAlmostEqual(resolved.threshold, 0.80 * 0.999, places=12)
        self.assertNotAlmostEqual(resolved.threshold, 0.80 - 0.001, places=12)

    def test_stage2_ppo_defaults_match_stage1_gtrxl_ppo_core(self):
        stage2_defaults = _class_literal_defaults("blb_stage2_rl/policy.py", "PPOConfig")
        seq_defaults = _class_literal_defaults(
            "blb_stage2_rl/sequential_policy.py",
            "SequentialPPOConfig",
        )

        self.assertEqual(stage2_defaults["lr"], 5e-5)
        self.assertEqual(stage2_defaults["clip_range"], 0.2)
        self.assertEqual(stage2_defaults["n_epochs"], 4)
        self.assertEqual(stage2_defaults["ent_coef"], 0.02)
        self.assertEqual(stage2_defaults["value_coef"], 0.5)
        self.assertEqual(stage2_defaults["max_grad_norm"], 0.5)

        self.assertEqual(seq_defaults["lr"], 5e-5)
        self.assertEqual(seq_defaults["clip_range"], 0.2)
        self.assertEqual(seq_defaults["n_epochs"], 4)
        self.assertEqual(seq_defaults["ent_coef"], 0.01)
        self.assertEqual(seq_defaults["value_coef"], 0.5)
        self.assertEqual(seq_defaults["max_grad_norm"], 0.5)
        self.assertEqual(seq_defaults["gamma"], 0.99)
        self.assertEqual(seq_defaults["gae_lambda"], 0.95)
        self.assertEqual(seq_defaults["value_clip_range"], 0.2)
        self.assertIs(seq_defaults["normalize_returns"], True)
        self.assertIs(seq_defaults["robust_advantage_norm"], False)
        self.assertIs(seq_defaults["use_kl_early_stop"], True)
        self.assertIs(seq_defaults["adaptive_lr_kl"], False)
        self.assertIs(seq_defaults["per_slot_entropy_recovery"], False)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required")
    def test_layerwise_terminal_credit_is_explicit_without_changing_legacy_defaults(self):
        from blb_stage2_rl.sequential_policy import SequentialPPOConfig

        legacy = SequentialPPOConfig()
        layerwise = SequentialPPOConfig(gamma=1.0, gae_lambda=1.0)

        self.assertEqual((legacy.gamma, legacy.gae_lambda), (0.99, 0.95))
        self.assertEqual((layerwise.gamma, layerwise.gae_lambda), (1.0, 1.0))


class Stage2Stage1RewardAlignmentTest(unittest.TestCase):
    def test_stage1_aligned_reward_gates_precision_then_stability_then_cost(self):
        reward_mod = _load_module_standalone("blb_stage2_rl/reward.py", "stage2_reward_alignment_a")
        BaselineCostStats = reward_mod.BaselineCostStats
        EpisodeMetrics = reward_mod.EpisodeMetrics
        RewardWeights = reward_mod.RewardWeights
        compute_reward = reward_mod.compute_reward

        class Signals:
            total_bits_sum = 1000
            total_fusion_count = 0
            any_invalid = False

        baseline = BaselineCostStats(
            total_bits_sum=1000,
            total_fusion_count=0,
            avg_k=13.0,
            loss_mean=0.350,
            loss_std=0.010,
            metric1_mean=0.880,
            metric2_mean=0.870,
            metric1_std=0.010,
            metric2_std=0.010,
        )
        metrics = EpisodeMetrics(
            loss_mean=0.346,
            loss_std=0.012,
            metric1_mean=0.881,
            metric2_mean=0.872,
            metric1_std=0.011,
            metric2_std=0.012,
        )
        weights = RewardWeights(
            reward_design="stage1_aligned",
            baseline_metric1=baseline.metric1_mean,
            baseline_metric2=baseline.metric2_mean,
            acc_tolerance=0.001,
            stab_tolerance=3.0,
            stab_floor=0.0,
        )

        got = compute_reward(
            metrics,
            Signals(),
            action_avg_k=11.0,
            baseline=baseline,
            weights=weights,
            external_cost_score=2.25,
            external_cost_rank=123.0,
        )

        loss_limit = baseline.loss_mean * 1.001
        m1_limit = baseline.metric1_mean * 0.999
        m2_limit = baseline.metric2_mean * 0.999
        stab_m1_limit = baseline.metric1_std * 3.0
        stab_m2_limit = baseline.metric2_std * 3.0
        stab_loss_limit = baseline.loss_std * 3.0
        precision_barriers = [
            _stage1_exact_barrier(metrics.loss_mean, loss_limit, upper=True),
            _stage1_exact_barrier(metrics.metric1_mean, m1_limit, upper=False),
            _stage1_exact_barrier(metrics.metric2_mean, m2_limit, upper=False),
        ]
        stability_barriers = [
            _stage1_exact_barrier(metrics.metric1_std, stab_m1_limit, upper=True),
            _stage1_exact_barrier(metrics.metric2_std, stab_m2_limit, upper=True),
            _stage1_exact_barrier(metrics.loss_std, stab_loss_limit, upper=True),
        ]
        expected = (
            sum(precision_barriers) / len(precision_barriers)
            + sum(stability_barriers) / len(stability_barriers)
            + 20.0 * 0.5
        ) / 20.0
        expected = max(-5.0, min(5.0, expected))

        self.assertAlmostEqual(got.reward, expected, places=7)
        self.assertEqual(got.priority, 3)
        self.assertTrue(got.metric_ok)
        self.assertTrue(got.stab_ok)
        self.assertAlmostEqual(got.cost_score, 0.5, places=7)

    def test_stage1_aligned_reward_ignores_cost_when_precision_is_violated(self):
        reward_mod = _load_module_standalone("blb_stage2_rl/reward.py", "stage2_reward_alignment_b")
        BaselineCostStats = reward_mod.BaselineCostStats
        EpisodeMetrics = reward_mod.EpisodeMetrics
        RewardWeights = reward_mod.RewardWeights
        compute_reward = reward_mod.compute_reward

        class Signals:
            total_bits_sum = 1000
            total_fusion_count = 0
            any_invalid = False

        baseline = BaselineCostStats(
            total_bits_sum=1000,
            avg_k=13.0,
            loss_mean=0.350,
            loss_std=0.010,
            metric1_mean=0.880,
            metric2_mean=0.870,
            metric1_std=0.010,
            metric2_std=0.010,
        )
        weights = RewardWeights(
            reward_design="stage1_aligned",
            baseline_metric1=baseline.metric1_mean,
            baseline_metric2=baseline.metric2_mean,
            acc_tolerance=0.001,
            stab_tolerance=3.0,
            stab_floor=0.0,
        )

        no_cost = compute_reward(
            EpisodeMetrics(
                loss_mean=0.346,
                loss_std=0.012,
                metric1_mean=0.870,
                metric2_mean=0.872,
                metric1_std=0.011,
                metric2_std=0.012,
            ),
            Signals(),
            action_avg_k=13.0,
            baseline=baseline,
            weights=weights,
            external_cost_score=0.0,
        )
        with_cost = compute_reward(
            EpisodeMetrics(
                loss_mean=0.346,
                loss_std=0.012,
                metric1_mean=0.870,
                metric2_mean=0.872,
                metric1_std=0.011,
                metric2_std=0.012,
            ),
            Signals(),
            action_avg_k=13.0,
            baseline=baseline,
            weights=weights,
            external_cost_score=2.25,
        )

        self.assertEqual(with_cost.priority, 1)
        self.assertEqual(with_cost.reward, no_cost.reward)
        self.assertEqual(with_cost.cost_score, 0.0)
        self.assertEqual(with_cost.cost_rank_score, 0.0)

    def test_stage1_aligned_reward_ignores_cost_until_stability_is_satisfied(self):
        reward_mod = _load_module_standalone("blb_stage2_rl/reward.py", "stage2_reward_alignment_stab")
        BaselineCostStats = reward_mod.BaselineCostStats
        EpisodeMetrics = reward_mod.EpisodeMetrics
        RewardWeights = reward_mod.RewardWeights
        compute_reward = reward_mod.compute_reward

        class Signals:
            total_bits_sum = 1000
            total_fusion_count = 0
            any_invalid = False

        baseline = BaselineCostStats(
            total_bits_sum=1000,
            avg_k=13.0,
            loss_mean=0.350,
            loss_std=0.010,
            metric1_mean=0.880,
            metric2_mean=0.870,
            metric1_std=0.010,
            metric2_std=0.010,
        )
        weights = RewardWeights(
            reward_design="stage1_aligned",
            baseline_metric1=baseline.metric1_mean,
            baseline_metric2=baseline.metric2_mean,
            acc_tolerance=0.001,
            stab_tolerance=3.0,
            stab_floor=0.0,
        )
        metrics = EpisodeMetrics(
            loss_mean=0.346,
            loss_std=0.050,
            metric1_mean=0.881,
            metric2_mean=0.872,
            metric1_std=0.050,
            metric2_std=0.050,
        )

        no_cost = compute_reward(
            metrics,
            Signals(),
            action_avg_k=13.0,
            baseline=baseline,
            weights=weights,
            external_cost_score=0.0,
        )
        with_cost = compute_reward(
            metrics,
            Signals(),
            action_avg_k=13.0,
            baseline=baseline,
            weights=weights,
            external_cost_score=2.25,
            external_cost_rank=123.0,
        )

        self.assertEqual(with_cost.priority, 2)
        self.assertEqual(with_cost.reward, no_cost.reward)
        self.assertEqual(with_cost.cost_score, 0.0)
        self.assertEqual(with_cost.cost_rank_score, 0.0)


class Stage2DenseRewardAlignmentTest(unittest.TestCase):
    def test_stage2_dense_step_reward_uses_stage1_positive_cost_saving(self):
        reward_mod = _load_module_standalone("blb_stage2_rl/reward.py", "stage2_reward_alignment_c")
        stage1_dense_cost_reward = reward_mod.stage1_dense_cost_reward

        self.assertAlmostEqual(stage1_dense_cost_reward(80, 100), 0.02)
        self.assertAlmostEqual(stage1_dense_cost_reward(100, 100), 0.0)
        self.assertAlmostEqual(stage1_dense_cost_reward(120, 100), 0.0)


if __name__ == "__main__":
    unittest.main()
