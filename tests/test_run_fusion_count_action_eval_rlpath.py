import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

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
