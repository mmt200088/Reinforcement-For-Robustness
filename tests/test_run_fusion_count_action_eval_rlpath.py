import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np

from json_utils import to_jsonable
from scripts.fusion_count_action_eval_common import (
    load_rlpath_action_configs,
    rlpath_config_group_key,
    rlpath_group_key,
    unique_rlpath_action_configs,
)


class NoCopyMapping:
    def __init__(self, payload):
        self.payload = dict(payload)

    def get(self, key, default=None):
        return self.payload.get(key, default)

    def __getitem__(self, key):
        return self.payload[key]

    def __iter__(self):
        raise AssertionError("unique config selection should not copy mappings")

    def __len__(self):
        raise AssertionError("unique config selection should not copy mappings")


class FusionCountActionEvalRLPathTest(unittest.TestCase):
    def test_fixed_map_option_converts_to_policy_local_index(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

        class FakeSeqEnv:
            def __init__(self):
                self.base = SimpleNamespace(probe_noise_seed=None)
                self._step_idx = 0
                self._schedule = [SimpleNamespace(
                    step_idx=0,
                    layer_idx=0,
                    block_idx=2,
                    graph_key_suffix="block2_mrpc",
                    map_option_ids=(1,),
                )]
                self.evaluated_actions = []

            def reset(self, *, seed):
                self._step_idx = 0

            def evaluate_step(self, action):
                self.evaluated_actions.append(list(action))
                return {
                    "valid": True,
                    "fusion_count": 1,
                    "boosted_field_values": {"output_truncation_k": 13},
                }

            def commit_step(self, _eval_info, *, defer_terminal_forward):
                self._step_idx = 1
                terminal_info = {
                    "metrics": SimpleNamespace(
                        loss_mean=0.3,
                        loss_std=0.01,
                        metric1_mean=0.88,
                        metric1_std=0.01,
                        metric2_mean=0.87,
                        metric2_std=0.01,
                    ),
                    "fusion_action_steps": [{
                        "block_idx": 2,
                        "fusion_count": 1,
                        "k_value": 13,
                        "graph_key": "block2_mrpc_L0",
                    }],
                }
                return np.zeros(1), 1.0, True, {"terminal_info": terminal_info}

        env = FakeSeqEnv()
        old_deps = rlpath._RUNTIME_DEPS
        try:
            rlpath._RUNTIME_DEPS = {"K_LEVELS": (8, 9, 11, 13, 10, 12)}
            result = rlpath._run_group(
                env,
                {
                    "name": "fixed_b2",
                    "path": Path("fixed_b2.json"),
                    "baseline_k_index": 3,
                    "group": {"option_by_graph": {"block2_mrpc": 1}},
                },
                seed=42,
            )
        finally:
            rlpath._RUNTIME_DEPS = old_deps

        self.assertEqual(env.evaluated_actions, [[0, 3]])
        self.assertEqual(result["step_records"][0]["policy_option_index"], 0)
        self.assertEqual(result["step_records"][0]["map_option_id"], 1)
        self.assertEqual(result["fusion_total"], 1)

    def test_module_import_is_dependency_light(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

        self.assertTrue(callable(rlpath.load_rlpath_action_configs))

    def test_load_action_configs_does_not_retain_full_payload(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = root / "candidate.json"
            config.write_text(
                json.dumps(
                    {
                        "baseline_k_index": 2,
                        "group": {
                            "name": "candidate",
                            "option_by_graph": {"block2_mrpc": 1},
                            "option_by_step": {"0": 1},
                        },
                        "large_unused_payload": [{"i": i} for i in range(64)],
                    }
                ),
                encoding="utf-8",
            )

            configs = load_rlpath_action_configs(root)

        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]["name"], "candidate")
        self.assertEqual(configs[0]["baseline_k_index"], 2)
        self.assertEqual(configs[0]["group"]["option_by_graph"], {"block2_mrpc": 1})
        self.assertNotIn("payload", configs[0])
        self.assertEqual(configs[0]["group_key"], rlpath_group_key(configs[0]))

    def test_load_action_configs_scans_directory_without_path_glob(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "_summary.json").write_text("{}", encoding="utf-8")
            (root / "._candidate.json").write_text("{}", encoding="utf-8")
            (root / "notes.txt").write_text("ignored", encoding="utf-8")
            (root / "candidate.json").write_text(
                json.dumps(
                    {
                        "baseline_k_index": 2,
                        "group": {"name": "candidate", "option_by_graph": {"block2_mrpc": 1}},
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(
                Path,
                "glob",
                side_effect=AssertionError("action config loading should not use Path.glob"),
            ):
                configs = load_rlpath_action_configs(root)

        self.assertEqual([cfg["name"] for cfg in configs], ["candidate"])

    def test_group_key_uses_retained_group_and_baseline_fields(self):
        left = {
            "name": "a",
            "group": {"option_by_graph": {"block2_mrpc": 1}, "option_by_step": {"0": 1}},
            "baseline_k_index": 2,
        }
        right = {
            "name": "b",
            "group": {"option_by_graph": {"block2_mrpc": 1}, "option_by_step": {"0": 1}},
            "baseline_k_index": 3,
        }

        self.assertNotEqual(rlpath_group_key(left), rlpath_group_key(right))

    def test_jsonable_reuses_json_native_nested_payloads(self):
        steps = [
            {"step_idx": i, "valid": bool(i % 2), "nested": {"fusion_count": i}}
            for i in range(8)
        ]

        converted = to_jsonable(steps, stringify_unknown=True, preserve_native=True)

        self.assertIs(converted, steps)
        self.assertIs(converted[0]["nested"], steps[0]["nested"])

    def test_jsonable_converts_only_branches_that_need_conversion(self):
        import numpy as np

        steps = [{"step_idx": i, "valid": True} for i in range(8)]
        payload = {"steps": steps, "array": np.array([1, 2, 3])}

        converted = to_jsonable(payload, stringify_unknown=True, preserve_native=True)

        self.assertIsNot(converted, payload)
        self.assertIs(converted["steps"], steps)
        self.assertEqual(converted["array"], [1, 2, 3])

    def test_jsonable_does_not_reconvert_changed_list_item(self):
        class CountedString:
            calls = 0

            def __str__(self):
                type(self).calls += 1
                return "converted"

        native = {"step_idx": 1, "valid": True}
        converted = to_jsonable([native, CountedString()], stringify_unknown=True, preserve_native=True)

        self.assertEqual(converted, [native, "converted"])
        self.assertIs(converted[0], native)
        self.assertEqual(CountedString.calls, 1)

    def test_jsonable_does_not_reconvert_changed_dict_item(self):
        class CountedString:
            calls = 0

            def __str__(self):
                type(self).calls += 1
                return "converted"

        native = {"step_idx": 1, "valid": True}
        converted = to_jsonable({"native": native, "custom": CountedString()}, stringify_unknown=True, preserve_native=True)

        self.assertEqual(converted, {"native": native, "custom": "converted"})
        self.assertIs(converted["native"], native)
        self.assertEqual(CountedString.calls, 1)

    def test_unique_configs_reuses_first_config_without_copying(self):
        first = NoCopyMapping({
            "name": "first",
            "group": {"option_by_graph": {"block2_mrpc": 1}, "option_by_step": {"0": 1}},
            "baseline_k_index": 2,
        })
        duplicate = NoCopyMapping({
            "name": "duplicate",
            "group": {"option_by_graph": {"block2_mrpc": 1}, "option_by_step": {"0": 1}},
            "baseline_k_index": 2,
        })

        unique = unique_rlpath_action_configs([first, duplicate])

        self.assertEqual(len(unique), 1)
        self.assertIs(next(iter(unique)), first)

    def test_unique_configs_reuses_cached_group_key(self):
        first = {"name": "first", "group_key": "same"}
        duplicate = {"name": "duplicate", "group_key": "same"}

        with mock.patch(
            "scripts.fusion_count_action_eval_common.rlpath_group_key",
            side_effect=AssertionError("cached group_key should be reused"),
        ):
            unique = unique_rlpath_action_configs([first, duplicate])

        self.assertEqual(len(unique), 1)
        self.assertIs(next(iter(unique)), first)

    def test_config_group_key_reuses_cached_group_key(self):
        self.assertEqual(rlpath_config_group_key({"group_key": "cached"}), "cached")

    def test_rlpath_script_has_no_local_common_wrappers(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

        source = Path(rlpath.__file__).read_text(encoding="utf-8")
        forbidden = [
            "def _resolve(",
            "def _json_int_list(",
            "def _iter_action_config_paths(",
            "def _load_action_configs(",
            "def _group_key(",
            "def _config_group_key(",
            "def _unique_configs(",
        ]
        for token in forbidden:
            self.assertNotIn(token, source)
        self.assertIn("resolve_repo_path", source)
        self.assertIn("unique_rlpath_action_configs", source)

    def test_main_streams_stdout_json_without_json_dumps_string(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

        source = Path(rlpath.__file__).read_text(encoding="utf-8")
        main_source = source[source.index("def main("):]

        self.assertIn("json.dump(", main_source)
        self.assertIn("sys.stdout", main_source)
        self.assertIn('sys.stdout.write("\\n")', main_source)
        self.assertNotIn("print(json.dumps(", main_source)

    def test_main_reuses_static_stage1_default_json_strings(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

        source = Path(rlpath.__file__).read_text(encoding="utf-8")
        main_source = source[source.index("def main("):]

        self.assertIn("DEFAULT_STAGE1_GELU_JSON = json.dumps(DEFAULT_STAGE1_GELU)", source)
        self.assertIn("DEFAULT_STAGE1_SOFTMAX_JSON = json.dumps(DEFAULT_STAGE1_SOFTMAX)", source)
        self.assertIn("default=DEFAULT_STAGE1_GELU_JSON", main_source)
        self.assertIn("default=DEFAULT_STAGE1_SOFTMAX_JSON", main_source)
        self.assertNotIn("default=json.dumps(DEFAULT_STAGE1_GELU)", main_source)
        self.assertNotIn("default=json.dumps(DEFAULT_STAGE1_SOFTMAX)", main_source)

    def test_main_streams_html_report_without_full_render_string_write(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

        source = Path(rlpath.__file__).read_text(encoding="utf-8")
        main_source = source[source.index("def main("):]

        self.assertTrue(hasattr(rlpath, "write_rendered_html"))
        self.assertTrue(hasattr(rlpath, "_HtmlPartsWriter"))
        self.assertIn("_HtmlPartsWriter(output_html)", source)
        self.assertNotIn("output_html.write_text(_render_html(combined)", main_source)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "parts.html"
            writer = rlpath._HtmlPartsWriter(path)
            writer.append("alpha")
            writer.extend(["beta", "gamma"])
            writer.close()

            self.assertEqual(path.read_text(encoding="utf-8"), "alpha\nbeta\ngamma")
