import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import run_fusion_count_action_eval as action_eval
from scripts.fusion_count_action_eval_common import (
    load_paean_action_configs,
    unique_paean_action_configs,
)


class NoCopyMapping:
    def __init__(self, payload):
        self.payload = dict(payload)

    def __getitem__(self, key):
        return self.payload[key]

    def __iter__(self):
        raise AssertionError("unique config selection should not copy mappings")

    def __len__(self):
        raise AssertionError("unique config selection should not copy mappings")

    def get(self, key, default=None):
        return self.payload.get(key, default)


class FusionCountActionEvalTest(unittest.TestCase):
    def test_load_action_configs_does_not_retain_full_payload(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = root / "candidate.json"
            config.write_text(
                json.dumps(
                    {
                        "group": {"name": "candidate", "family": "bench"},
                        "action_vec": [1, 2, 3],
                        "large_unused_payload": [{"i": i} for i in range(32)],
                    }
                ),
                encoding="utf-8",
            )

            configs = load_paean_action_configs(root)

        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]["name"], "candidate")
        self.assertEqual(configs[0]["group"], {"name": "candidate", "family": "bench"})
        self.assertIn("action_hash", configs[0])
        self.assertNotIn("payload", configs[0])

    def test_load_action_configs_scans_directory_without_path_glob(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "_summary.json").write_text("{}", encoding="utf-8")
            (root / "._candidate.json").write_text("{}", encoding="utf-8")
            (root / "notes.txt").write_text("ignored", encoding="utf-8")
            (root / "candidate.json").write_text(
                json.dumps({"group": {"name": "candidate"}, "action_vec": [1, 2, 3]}),
                encoding="utf-8",
            )

            with mock.patch.object(
                Path,
                "glob",
                side_effect=AssertionError("action config loading should not use Path.glob"),
            ):
                configs = load_paean_action_configs(root)

        self.assertEqual([cfg["name"] for cfg in configs], ["candidate"])

    def test_build_combined_avoids_deepcopying_large_candidate_payloads(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output_root = root / "paean"
            result_dir = output_root / "mrpc" / "rl" / "canonical" / "final_eval"
            result_dir.mkdir(parents=True)
            result_path = result_dir / "blb_action_final_eval_results_mrpc.json"
            result_path.write_text(
                json.dumps(
                    {
                        "baseline": {"loss": 1.0, "p": 0.8, "s": 0.7, "evaluation_n": 5},
                        "candidate_results": [
                            {
                                "loss": 0.9,
                                "p": 0.81,
                                "s": 0.71,
                                "evaluation_n": 5,
                                "large_diagnostics": [{"value": i} for i in range(16)],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            configs = [
                {
                    "name": "canonical",
                    "path": root / "canonical.json",
                    "action_hash": "same-hash",
                    "group": {"name": "canonical"},
                },
                {
                    "name": "alias",
                    "path": root / "alias.json",
                    "action_hash": "same-hash",
                    "group": {"name": "alias"},
                },
            ]

            with mock.patch(
                "copy.deepcopy",
                side_effect=AssertionError("candidate cloning should be shallow"),
            ):
                combined = action_eval._build_combined(
                    configs=configs,
                    output_root=output_root,
                    map_report={"profile": "mrpc"},
                    stage1_gelu=[1] * 12,
                    stage1_softmax=[6] * 12,
                )

        self.assertEqual(combined["evaluation_protocol"]["unique_action_runs"], 1)
        self.assertEqual([row["name"] for row in combined["group_results"]], ["canonical", "alias"])
        self.assertTrue(combined["group_results"][1]["reused_from_canonical"])
        self.assertIn("large_diagnostics", combined["group_results"][0])

    def test_unique_configs_reuses_first_config_without_copying(self):
        first = NoCopyMapping({
            "name": "first",
            "path": Path("first.json"),
            "action_hash": "same",
            "group": {"name": "first"},
        })
        duplicate = NoCopyMapping({
            "name": "duplicate",
            "path": Path("duplicate.json"),
            "action_hash": "same",
            "group": {"name": "duplicate"},
        })

        unique = unique_paean_action_configs([first, duplicate])

        self.assertEqual(len(unique), 1)
        self.assertIs(next(iter(unique)), first)

    def test_action_eval_script_has_no_local_common_wrappers(self):
        source = Path(action_eval.__file__).read_text(encoding="utf-8")
        forbidden = [
            "def _resolve(",
            "def _json_int_list(",
            "def _json_hash(",
            "def _iter_action_config_paths(",
            "def _load_action_configs(",
            "def _unique_configs(",
        ]
        for token in forbidden:
            self.assertNotIn(token, source)
        self.assertIn("resolve_repo_path", source)
        self.assertIn("unique_paean_action_configs", source)
        self.assertIn("from runtime_error_reporter import format_command", source)
        self.assertNotIn('" ".join(cmd)', source)

    def test_main_streams_stdout_json_without_json_dumps_string(self):
        source = Path(action_eval.__file__).read_text(encoding="utf-8")
        main_source = source[source.index("def main("):]

        self.assertIn("json.dump(", main_source)
        self.assertIn("sys.stdout", main_source)
        self.assertIn('sys.stdout.write("\\n")', main_source)
        self.assertNotIn("print(json.dumps(", main_source)

    def test_main_reuses_static_stage1_default_json_strings(self):
        source = Path(action_eval.__file__).read_text(encoding="utf-8")
        run_one_source = source[source.index("def _run_one("):source.index("def _load_result(")]
        main_source = source[source.index("def main("):]

        self.assertIn("DEFAULT_STAGE1_GELU_JSON = json.dumps(DEFAULT_STAGE1_GELU)", source)
        self.assertIn("DEFAULT_STAGE1_SOFTMAX_JSON = json.dumps(DEFAULT_STAGE1_SOFTMAX)", source)
        self.assertIn('DEFAULT_MANUAL_NOISE_JSON = json.dumps(DEFAULT_MANUAL_NOISE, separators=(",", ":"))', source)
        self.assertIn("default=DEFAULT_STAGE1_GELU_JSON", main_source)
        self.assertIn("default=DEFAULT_STAGE1_SOFTMAX_JSON", main_source)
        self.assertIn("DEFAULT_MANUAL_NOISE_JSON", run_one_source)
        self.assertNotIn("default=json.dumps(DEFAULT_STAGE1_GELU)", main_source)
        self.assertNotIn("default=json.dumps(DEFAULT_STAGE1_SOFTMAX)", main_source)
        self.assertNotIn("json.dumps(DEFAULT_MANUAL_NOISE", run_one_source)


if __name__ == "__main__":
    unittest.main()
