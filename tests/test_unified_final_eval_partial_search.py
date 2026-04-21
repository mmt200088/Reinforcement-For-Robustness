import json
import tempfile
import unittest
from pathlib import Path


class _DummyEvaluator:
    dataset_key = "mrpc"
    model_type = "bert-base"
    total_layers = 4
    INPUT_NOISE_COST_MAP = {sf: sf * 0.025 for sf in (22, 24, 26, 28, 30)}
    WEIGHT_NOISE_COST_MAP = {sf: sf * 0.025 for sf in (14, 16, 18, 20, 22)}
    WFFN1_NOISE_COST_MAP = {sf: sf * 0.025 for sf in (16, 18, 20, 22, 24)}

    def __init__(self):
        self.logs = []

    def log(self, message):
        self.logs.append(str(message))


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


if __name__ == "__main__":
    unittest.main()
