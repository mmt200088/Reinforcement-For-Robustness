import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from json_utils import to_jsonable


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

        self.assertTrue(callable(rlpath._load_action_configs))

    def test_load_action_configs_does_not_retain_full_payload(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

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

            configs = rlpath._load_action_configs(root)

        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]["name"], "candidate")
        self.assertEqual(configs[0]["baseline_k_index"], 2)
        self.assertEqual(configs[0]["group"]["option_by_graph"], {"block2_mrpc": 1})
        self.assertNotIn("payload", configs[0])
        self.assertEqual(configs[0]["group_key"], rlpath._group_key(configs[0]))

    def test_load_action_configs_scans_directory_without_path_glob(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

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
                configs = rlpath._load_action_configs(root)

        self.assertEqual([cfg["name"] for cfg in configs], ["candidate"])

    def test_group_key_uses_retained_group_and_baseline_fields(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

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

        self.assertNotEqual(rlpath._group_key(left), rlpath._group_key(right))

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
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

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

        unique = rlpath._unique_configs([first, duplicate])

        self.assertEqual(len(unique), 1)
        self.assertIs(next(iter(unique)), first)

    def test_unique_configs_reuses_cached_group_key(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

        first = {"name": "first", "group_key": "same"}
        duplicate = {"name": "duplicate", "group_key": "same"}

        with mock.patch.object(
            rlpath,
            "_group_key",
            side_effect=AssertionError("cached group_key should be reused"),
        ):
            unique = rlpath._unique_configs([first, duplicate])

        self.assertEqual(len(unique), 1)
        self.assertIs(next(iter(unique)), first)

    def test_loader_wrapper_delegates_to_shared_common_helper(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

        with mock.patch.object(
            rlpath,
            "load_rlpath_action_configs",
            return_value=[{"name": "shared"}],
        ) as helper:
            configs = rlpath._load_action_configs(Path("actions"))

        helper.assert_called_once_with(Path("actions"))
        self.assertEqual(configs, [{"name": "shared"}])
