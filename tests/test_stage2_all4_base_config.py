from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from blb_stage2_rl.action_space import step_schedule
from layer_importance_evaluator import LayerImportanceEvaluator


REPO_ROOT = Path(__file__).resolve().parents[1]


class _Stage1ResultResolver:
    config_source = "search"

    def __init__(self) -> None:
        self.calls = 0

    def resolve_stage1_only(self, *, search_best_stage1, total_layers):
        self.calls += 1
        return (
            np.asarray(search_best_stage1["gelu"], dtype=int),
            np.asarray(search_best_stage1["softmax"], dtype=int),
            "search",
        )


def _bare_evaluator(source: str) -> LayerImportanceEvaluator:
    ev = LayerImportanceEvaluator.__new__(LayerImportanceEvaluator)
    ev.total_layers = 12
    ev.decoupled_layout = False
    ev.stage2_fixed_config_source = source
    ev.stage2_fixed_config_path = ""
    ev.stage2_manual_gelu = None
    ev.stage2_manual_softmax = None
    ev.log = lambda *_args, **_kwargs: None
    return ev


class Stage2All4BaseConfigTest(unittest.TestCase):
    def test_large_mrpc_stage1_record_resolves_through_production_stage2_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            record_dir = root / "stage1" / "record" / "bert large mrpc 1 20260725"
            record_dir.mkdir(parents=True)
            expected_gelu = [
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            ]
            (record_dir / "final_config.json").write_text(
                json.dumps({
                    "gelu_degree_per_layer": expected_gelu,
                    "softmax_degree_per_layer": [6] * 24,
                }),
                encoding="utf-8",
            )

            ev = _bare_evaluator("stage1_result")
            ev.total_layers = 24
            ev.decoupled_layout = True
            ev.run_output_dir = str(root / "stage2" / "bert large mrpc")
            ev.stage1_run_id = ""
            ev._build_stage2_fixed_config_resolver = lambda: object()

            gelu, softmax, label, source = ev._resolve_stage2_fixed_stage1_config()

        self.assertEqual(gelu.tolist(), expected_gelu)
        self.assertEqual(softmax.tolist(), [6] * 24)
        self.assertIn("stage1_record:bert large mrpc", source)
        self.assertIn("softmax fixed deg6", label)

    def test_all4_source_resolves_without_final_eval_fallback(self):
        ev = _bare_evaluator("all4")

        def forbidden_final_eval_resolver():
            raise AssertionError("all4 must not reuse the final-eval resolver")

        ev._build_final_eval_runner = forbidden_final_eval_resolver

        gelu, softmax, label, source = ev._resolve_stage2_fixed_stage1_config()

        self.assertEqual(gelu.tolist(), [4] * 12)
        self.assertEqual(softmax.tolist(), [6] * 12)
        self.assertEqual(source, "stage2_all4")
        self.assertIn("all4", label.lower())

    def test_stage1_result_source_remains_selectable(self):
        ev = _bare_evaluator("stage1_result")
        resolver = _Stage1ResultResolver()
        ev._build_stage2_fixed_config_resolver = lambda: resolver

        def forbidden_final_eval_resolver():
            raise AssertionError("Stage-2 must use its dedicated resolver inputs")

        ev._build_final_eval_runner = forbidden_final_eval_resolver
        searched = {
            "gelu": np.asarray([1, 2] + [1] * 10, dtype=int),
            "softmax": np.asarray([6] * 12, dtype=int),
        }

        gelu, softmax, _label, source = ev._resolve_stage2_fixed_stage1_config(
            search_best_config=searched
        )

        self.assertEqual(gelu.tolist(), searched["gelu"].tolist())
        self.assertEqual(softmax.tolist(), [6] * 12)
        self.assertEqual(source, "search")
        self.assertEqual(resolver.calls, 1)

    def test_all4_schedule_selects_block5_n4_in_every_layer(self):
        schedule = step_schedule(
            12,
            profile="mrpc",
            attn_degree_per_layer=[6] * 12,
            gelu_degree_per_layer=[4] * 12,
        )
        block5 = [step for step in schedule if step.block_idx == 5]

        self.assertEqual(len(block5), 12)
        self.assertEqual({step.graph_key_suffix for step in block5}, {"block5_n4"})

    def test_all_committed_profiles_have_degree4_block5_map(self):
        maps_root = REPO_ROOT / "blb_stage2_rl" / "fusion_maps"
        profiles = ("mrpc", "rte", "sst2", "mrpc_large", "rte_large", "sst2_large")

        for profile in profiles:
            with self.subTest(profile=profile):
                path = maps_root / profile / "block5_n4.json"
                self.assertTrue(path.is_file(), msg=str(path))
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(payload["graph_key"], "block5_n4")
                self.assertEqual(payload["gelu_degree"], 4)

    def test_rl_tune_forwards_dedicated_stage2_config(self):
        source = (REPO_ROOT / "rl_tune.py").read_text(encoding="utf-8")
        call = source[source.index("importance_evaluator = LayerImportanceEvaluator(") :]
        required = (
            "stage2_fixed_config_source=stage2_fixed_config_source",
            "stage2_fixed_config_path=stage2_fixed_config_path",
            "stage2_manual_gelu=parsed_stage2_manual_gelu",
            "stage2_manual_softmax=parsed_stage2_manual_softmax",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, call)

if __name__ == "__main__":
    unittest.main()
