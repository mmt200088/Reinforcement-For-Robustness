from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import final_evaluation_module as fem


class FinalEvaluationConfigCacheTest(unittest.TestCase):
    def _make_runner(self, config_path: Path):
        runner = fem.UnifiedFinalEvaluationModule.__new__(fem.UnifiedFinalEvaluationModule)
        runner.config_path = str(config_path)
        runner.evaluator = SimpleNamespace(
            total_layers=2,
            model_type="bert-base",
            dataset_key="mrpc",
            model=None,
            log=lambda *_args, **_kwargs: None,
        )
        runner.allowed_gelu_selected = [1, 2, 4]
        runner.allowed_softmax = [6]
        runner.input_noise_allowed = [1]
        runner.weight_noise_allowed = [1]
        runner.wffn1_noise_allowed = [1]
        return runner

    def test_stage1_and_stage2_json_resolution_share_loaded_config(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "final_eval_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "_comment": "ignored",
                        "bert-base": {
                            "mrpc": {
                                "stage1": {"gelu": [1, 2], "softmax": [6, 6]},
                                "stage2": {
                                    key: [1, 1]
                                    for key in fem.NOISE_SCALING_FACTOR_KEYS
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            runner = self._make_runner(config_path)
            original_load = fem.json.load
            load_count = 0

            def counting_load(handle):
                nonlocal load_count
                load_count += 1
                return original_load(handle)

            with mock.patch.object(fem.json, "load", side_effect=counting_load):
                runner._resolve_stage1_from_json(total_layers=2)
                runner._resolve_stage2_from_json(total_layers=2)

            self.assertEqual(load_count, 1)


if __name__ == "__main__":
    unittest.main()
