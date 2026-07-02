import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import run_fusion_count_action_eval as action_eval


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

            configs = action_eval._load_action_configs(root)

        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]["name"], "candidate")
        self.assertEqual(configs[0]["group"], {"name": "candidate", "family": "bench"})
        self.assertIn("action_hash", configs[0])
        self.assertNotIn("payload", configs[0])

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


if __name__ == "__main__":
    unittest.main()
