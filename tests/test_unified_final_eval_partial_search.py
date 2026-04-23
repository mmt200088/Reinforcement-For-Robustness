import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class _DummyEvaluator:
    dataset_key = "mrpc"
    model_type = "bert-base"
    total_layers = 4
    GELU_COST_MAP = {4: 3.0, 2: 2.5, 1: 1.0, 0: -1.0}
    SOFTMAX_COST_MAP = {6: 3.0, 5: 2.5, 4: 2.0, 3: 1.5, 2: 1.0}
    INPUT_NOISE_COST_MAP = {sf: sf * 0.025 for sf in (22, 24, 26, 28, 30)}
    WEIGHT_NOISE_COST_MAP = {sf: sf * 0.025 for sf in (14, 16, 18, 20, 22)}
    WFFN1_NOISE_COST_MAP = {sf: sf * 0.025 for sf in (16, 18, 20, 22, 24)}

    def __init__(self):
        self.logs = []

    def log(self, message):
        self.logs.append(str(message))


class _RunEvaluator(_DummyEvaluator):
    total_layers = 2

    def __init__(self):
        super().__init__()
        self.clean_eval_calls = 0
        self.clean_repeat_calls = 0
        self.noisy_single_calls = 0
        self.noisy_repeat_calls = 0

    def get_num_metrics(self):
        return 2

    def get_metric_short_names(self):
        return ["Acc", "F1"]

    def build_constraint_limits_from_metrics(self, base_loss, base_p, base_s):
        return {
            "loss": float(base_loss) * 1.1,
            "metric1": float(base_p) * 0.99,
            "metric2": float(base_s) * 0.99,
        }

    def evaluate_model(self, gelu, softmax, use_train=True, split=None):
        self.clean_eval_calls += 1
        self.last_clean_split = split
        return 0.123, 0.8799, 0.8774, 1.0

    def evaluate_model_repeated(self, *args, **kwargs):
        self.clean_repeat_calls += 1
        raise AssertionError("clean baseline should not use repeated evaluation")

    def evaluate_model_with_attention_noise(self, *args, **kwargs):
        self.noisy_single_calls += 1
        return 0.333, 0.8, 0.79, 2.0

    def evaluate_model_with_attention_noise_repeated(self, *args, repeats=1, **kwargs):
        self.noisy_repeat_calls += 1
        trials = [
            {"loss": 0.3 + i * 0.01, "p": 0.8 + i * 0.001, "s": 0.79 + i * 0.001, "time_ms": 2.0}
            for i in range(int(repeats))
        ]
        return {
            "split_name": "validation_full",
            "n": int(repeats),
            "loss_mean": float(np.mean([t["loss"] for t in trials])),
            "loss_std": float(np.std([t["loss"] for t in trials])),
            "p_mean": float(np.mean([t["p"] for t in trials])),
            "p_std": float(np.std([t["p"] for t in trials])),
            "s_mean": float(np.mean([t["s"] for t in trials])),
            "s_std": float(np.std([t["s"] for t in trials])),
            "time_mean_ms": 2.0,
            "trials": trials,
        }

    def get_simulated_cost(self, gelu, softmax):
        g_c = sum(self.GELU_COST_MAP[int(d)] for d in gelu)
        s_c = sum(self.SOFTMAX_COST_MAP[int(d)] for d in softmax)
        return g_c + s_c, g_c, s_c

    def get_noise_simulated_cost(self, **noise_cfg):
        from final_evaluation_module import BREAKDOWN_KEYS, SHORT_KEY_TO_FULL

        breakdown = {}
        for short in BREAKDOWN_KEYS:
            values = np.asarray(noise_cfg[SHORT_KEY_TO_FULL[short]], dtype=int)
            if short == "x":
                cost_map = self.INPUT_NOISE_COST_MAP
            elif short == "wffn1":
                cost_map = self.WFFN1_NOISE_COST_MAP
            else:
                cost_map = self.WEIGHT_NOISE_COST_MAP
            breakdown[short] = float(sum(cost_map[int(v)] for v in values))
        return float(sum(breakdown.values())), breakdown

    def apply_configuration(self, *args, **kwargs):
        pass

    def clear_input_noise_configuration(self):
        pass

    def clear_weight_noise_configuration(self):
        pass


class _NoPlotFinalEvalModuleMixin:
    def _save_results_json(self, **kwargs):
        self.saved_results_payload = kwargs
        return str(Path(self.results_dir) / f"final_eval_results_{self.evaluator.dataset_key}.json")

    def _plot_results(self, *args, **kwargs):
        return None

    def _log_performance_table(self, *args, **kwargs):
        pass

    def _log_random_summary(self, *args, **kwargs):
        pass


class UnifiedFinalEvalPartialSearchTests(unittest.TestCase):
    def _build_module_and_config(self):
        try:
            from final_evaluation_module import (
                BREAKDOWN_KEYS,
                SHORT_KEY_TO_FULL,
                UnifiedFinalEvaluationModule,
            )
        except ImportError as exc:
            self.skipTest(f"final_evaluation_module import unavailable: {exc}")

        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        tmp.close()
        cfg_path = Path(tmp.name)

        evaluator = _DummyEvaluator()
        module = UnifiedFinalEvaluationModule(
            evaluator=evaluator,
            config_source="search",
            config_path=str(cfg_path),
            random_seed=42,
            permutation_trials=1,
            cost_equivalent_trials=1,
            budget_equivalent_trials=1,
            stage1_budget_trials=1,
            stage2_budget_trials=1,
            repeat_n=1,
        )

        stage1_cfg = {
            "gelu": [4, 4, 4, 4],
            "softmax": [6, 6, 6, 6],
        }
        stage2_cfg = {}
        for short_key in BREAKDOWN_KEYS:
            full_key = SHORT_KEY_TO_FULL[short_key]
            allowed = module._stage2_allowed(short_key)
            stage2_cfg[short_key] = [int(list(allowed)[0])] * evaluator.total_layers
            # Duplicate full keys to keep backward compatibility with old json variants.
            stage2_cfg[full_key] = list(stage2_cfg[short_key])

        config = {
            "bert-base": {
                "mrpc": {
                    "stage1": stage1_cfg,
                    "stage2": stage2_cfg,
                }
            }
        }
        cfg_path.write_text(json.dumps(config), encoding="utf-8")
        return module, cfg_path, stage1_cfg

    def test_search_source_falls_back_stage1_to_json(self):
        module, cfg_path, stage1_cfg = self._build_module_and_config()
        self.addCleanup(lambda: cfg_path.unlink(missing_ok=True))

        stage2_search = {}
        for key in (
            "input_noise_scaling_factors",
            "wq_noise_scaling_factors",
            "wk_noise_scaling_factors",
            "wv_noise_scaling_factors",
            "wo_noise_scaling_factors",
            "wffn1_noise_scaling_factors",
            "wffn2_noise_scaling_factors",
        ):
            stage2_search[key] = [int(max(module._stage2_allowed(module._full_to_short(key))))] * 4

        gelu, softmax, noise_cfg, source = module._resolve_selected_config(
            search_best_stage1=None,
            search_best_stage2=stage2_search,
            total_layers=4,
        )

        self.assertEqual(gelu.tolist(), stage1_cfg["gelu"])
        self.assertEqual(softmax.tolist(), stage1_cfg["softmax"])
        self.assertIn("stage1=json", source)
        self.assertIn("stage2=search", source)
        self.assertEqual(noise_cfg["wq_noise_scaling_factors"].tolist(), stage2_search["wq_noise_scaling_factors"])

    def test_search_source_falls_back_stage2_to_json(self):
        module, cfg_path, _ = self._build_module_and_config()
        self.addCleanup(lambda: cfg_path.unlink(missing_ok=True))

        stage1_search = {
            "gelu": [4, 4, 4, 4],
            "softmax": [6, 6, 6, 6],
        }
        gelu, softmax, _, source = module._resolve_selected_config(
            search_best_stage1=stage1_search,
            search_best_stage2=None,
            total_layers=4,
        )

        self.assertEqual(gelu.tolist(), stage1_search["gelu"])
        self.assertEqual(softmax.tolist(), stage1_search["softmax"])
        self.assertIn("stage1=search", source)
        self.assertIn("stage2=json", source)

    def test_resolve_stage1_only_allows_json_fallback_in_search_mode(self):
        module, cfg_path, stage1_cfg = self._build_module_and_config()
        self.addCleanup(lambda: cfg_path.unlink(missing_ok=True))

        gelu, softmax, source = module.resolve_stage1_only(
            search_best_stage1=None,
            total_layers=4,
        )

        self.assertEqual(source, "json")
        self.assertEqual(gelu.tolist(), stage1_cfg["gelu"])
        self.assertEqual(softmax.tolist(), stage1_cfg["softmax"])

    def test_search_source_still_rejects_both_stage_results_missing(self):
        module, cfg_path, _ = self._build_module_and_config()
        self.addCleanup(lambda: cfg_path.unlink(missing_ok=True))

        with self.assertRaisesRegex(ValueError, "at least one search result"):
            module._resolve_selected_config(
                search_best_stage1=None,
                search_best_stage2=None,
                total_layers=4,
            )

    def test_stage2_budget_sampler_matches_target_cost_exactly(self):
        module, cfg_path, _ = self._build_module_and_config()
        self.addCleanup(lambda: cfg_path.unlink(missing_ok=True))

        import numpy as np

        rng = np.random.default_rng(7)
        target_cfg = {}
        from final_evaluation_module import BREAKDOWN_KEYS, SHORT_KEY_TO_FULL

        for short_key in BREAKDOWN_KEYS:
            full_key = SHORT_KEY_TO_FULL[short_key]
            allowed = list(module._stage2_allowed(short_key))
            target_cfg[full_key] = np.array(
                [allowed[0], allowed[-1], allowed[1], allowed[-2]],
                dtype=int,
            )

        target_key = module._stage2_config_cost_key(target_cfg)
        sampled = module._sample_stage2_total_cost(rng, target_key / 40.0, 4)

        self.assertIsNotNone(sampled)
        self.assertEqual(module._stage2_config_cost_key(sampled), target_key)

    def test_final_eval_baseline_is_single_eval_when_noisy_groups_repeat(self):
        from final_evaluation_module import (
            NOISE_SCALING_FACTOR_KEYS,
            UnifiedFinalEvaluationModule,
        )

        class NoPlotModule(_NoPlotFinalEvalModuleMixin, UnifiedFinalEvaluationModule):
            pass

        evaluator = _RunEvaluator()
        noise_cfg = {
            key: [30 if key.startswith("input") else 22] * evaluator.total_layers
            for key in NOISE_SCALING_FACTOR_KEYS
        }
        noise_cfg["wffn1_noise_scaling_factors"] = [24] * evaluator.total_layers
        with tempfile.TemporaryDirectory() as tmpdir:
            module = NoPlotModule(
                evaluator=evaluator,
                config_source="manual",
                manual_stage1_gelu=[4, 4],
                manual_stage1_softmax=[6, 6],
                manual_stage2_noise=noise_cfg,
                permutation_trials=0,
                cost_equivalent_trials=0,
                budget_equivalent_trials=0,
                stage1_budget_trials=0,
                stage2_budget_trials=0,
                repeat_n=5,
                results_dir=tmpdir,
            )
            result = module.run(
                search_best_stage1=None,
                search_best_stage2=None,
                baseline_stage1_gelu=np.array([4, 4], dtype=int),
                baseline_stage1_softmax=np.array([6, 6], dtype=int),
                baseline_noise_tot_c=1.0,
                limit_loss=1.0,
                limit_p=0.0,
                limit_s=0.0,
            )

        self.assertEqual(evaluator.clean_eval_calls, 1)
        self.assertEqual(evaluator.clean_repeat_calls, 0)
        self.assertEqual(evaluator.last_clean_split, "validation_full")
        self.assertIsNone(result["baseline_repeat"])
        self.assertNotIn("evaluation_n", result["baseline_result"])
        self.assertEqual(evaluator.noisy_repeat_calls, 1)
        self.assertEqual(evaluator.noisy_single_calls, 0)
        self.assertEqual(result["optimized_result"]["evaluation_n"], 5)

    def test_random_groups_request_repeat_evaluation(self):
        from final_evaluation_module import (
            NOISE_SCALING_FACTOR_KEYS,
            UnifiedFinalEvaluationModule,
        )

        module, cfg_path, _ = self._build_module_and_config()
        self.addCleanup(lambda: cfg_path.unlink(missing_ok=True))
        module.stage1_budget_trials = 0
        module.stage2_budget_trials = 0
        module.permutation_trials = 3
        module.cost_equivalent_trials = 0
        module.budget_equivalent_trials = 0
        module.repeat_n = 4

        opt_gelu = np.array([1, 2, 4, 4], dtype=int)
        opt_softmax = np.array([2, 3, 5, 6], dtype=int)
        opt_noise_cfg = {
            key: np.array([22, 24, 28, 30], dtype=int)
            for key in NOISE_SCALING_FACTOR_KEYS
        }
        for key in (
            "wq_noise_scaling_factors",
            "wk_noise_scaling_factors",
            "wv_noise_scaling_factors",
            "wo_noise_scaling_factors",
            "wffn2_noise_scaling_factors",
        ):
            opt_noise_cfg[key] = np.array([14, 16, 20, 22], dtype=int)
        opt_noise_cfg["wffn1_noise_scaling_factors"] = np.array(
            [16, 18, 22, 24], dtype=int
        )
        opt_stage1_tot_c = 18.5
        opt_stage2_tot_c = 14.4
        opt_breakdown = {short: 1.0 for short in ("x", "wq", "wk", "wv", "wo", "wffn1", "wffn2")}

        want_repeat_flags = []

        def build_result(name, family, gelu, softmax, noise_cfg, want_repeat=False):
            want_repeat_flags.append(bool(want_repeat))
            return {"name": name, "family": family}, {"stats": {"n": module.repeat_n}, "trials": []}

        results = module._generate_random_results(
            opt_gelu=opt_gelu,
            opt_softmax=opt_softmax,
            opt_noise_cfg=opt_noise_cfg,
            opt_stage1_tot_c=opt_stage1_tot_c,
            opt_stage2_tot_c=opt_stage2_tot_c,
            opt_breakdown=opt_breakdown,
            total_layers=4,
            build_result=build_result,
        )

        self.assertGreater(len(results), 0)
        self.assertTrue(want_repeat_flags)
        self.assertTrue(all(want_repeat_flags))
        self.assertTrue(all("repeat_evaluation" in result for result in results))


if __name__ == "__main__":
    unittest.main()
