import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


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
