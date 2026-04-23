import json
import tempfile
import unittest
from pathlib import Path


class CompareRunnerUnifiedFinalTests(unittest.TestCase):
    def test_payload_uses_optimized_from_unified_final_eval(self):
        from rl_ga_compare_runner import build_stage_compare_payload

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rl_json = root / "rl_final.json"
            ga_json = root / "ga_final.json"

            def write_result(path: Path, loss: float, p: float, stage2_cost: float) -> None:
                payload = {
                    "dataset": "mrpc",
                    "selected_source": "json",
                    "evaluation_protocol": {
                        "version": 2,
                        "baseline": "single_clean_validation_full",
                        "noisy_repeat_n": 3,
                        "random_groups": "disabled",
                    },
                    "baseline": {
                        "loss": 1.0,
                        "p": 0.1,
                        "s": 0.1,
                        "stage1_tot_c": 72.0,
                        "stage2_tot_c": 0.0,
                        "show_cost_as_na": True,
                        "gelu": [4] * 12,
                        "softmax": [6] * 12,
                    },
                    "optimized_stage1": {
                        "gelu": [1] * 12,
                        "softmax": [2] * 12,
                    },
                    "optimized": {
                        "loss": loss,
                        "p": p,
                        "s": p,
                        "time_ms": 12.0,
                        "stage1_tot_c": 37.0,
                        "stage2_tot_c": stage2_cost,
                        "gelu": [1] * 12,
                        "softmax": [2] * 12,
                        "noise_config": {
                            "input_noise_scaling_factors": [22] * 12,
                        },
                        "feasible": True,
                    },
                    "optimized_repeat_evaluation": {
                        "stats": {
                            "n": 3,
                            "loss_mean": loss,
                            "loss_std": 0.01,
                            "p_mean": p,
                            "p_std": 0.02,
                            "s_mean": p,
                            "s_std": 0.02,
                        },
                        "trials": [],
                    },
                }
                path.write_text(json.dumps(payload), encoding="utf-8")

            write_result(rl_json, 0.2, 0.9, 42.0)
            write_result(ga_json, 0.3, 0.8, 43.0)

            payload = build_stage_compare_payload(
                stage_label="final",
                dataset="mrpc",
                compare_root=root,
                rl_run_dir=root / "rl",
                ga_run_dir=root / "ga",
                rl_json_path=rl_json,
                ga_json_path=ga_json,
                rl_warnings=[],
                ga_warnings=[],
                process_meta={
                    "compare_warnings": [],
                    "rl": {"state": "completed", "return_code": 0},
                    "ga": {"state": "completed", "return_code": 0},
                },
            )

            self.assertEqual(payload["sides"]["rl"]["selected_origin"], "optimized")
            self.assertEqual(payload["sides"]["rl"]["selected"]["loss"], 0.2)
            self.assertEqual(payload["sides"]["rl"]["selected"]["stage2_tot_c"], 42.0)
            self.assertEqual(
                payload["sides"]["rl"]["stage1_selected_source"],
                "final_eval_optimized_stage1",
            )
            self.assertIsNotNone(payload["stage2_repeat_summary"])

    def test_final_eval_protocol_rejects_repeated_baseline_json(self):
        from rl_ga_compare_runner import final_eval_json_matches_protocol

        stale = {
            "evaluation_protocol": {
                "version": 2,
                "baseline": "single_clean_validation_full",
                "noisy_repeat_n": 5,
                "random_groups": "enabled",
            },
            "baseline": {"loss": 0.1, "p": 0.9, "s": 0.9},
            "baseline_repeat_evaluation": {"stats": {"n": 5}},
            "optimized": {"evaluation_n": 5},
            "random_results": [{"evaluation_n": 5}],
        }
        self.assertFalse(
            final_eval_json_matches_protocol(
                stale,
                repeat_n=5,
                expect_random_groups=True,
            )
        )

    def test_final_eval_protocol_requires_random_repeats(self):
        from rl_ga_compare_runner import final_eval_json_matches_protocol

        current = {
            "evaluation_protocol": {
                "version": 2,
                "baseline": "single_clean_validation_full",
                "noisy_repeat_n": 5,
                "random_groups": "enabled",
            },
            "baseline": {"loss": 0.1, "p": 0.9, "s": 0.9},
            "optimized": {"evaluation_n": 5},
            "random_results": [{"evaluation_n": 5}],
        }
        missing_random_repeat = {
            **current,
            "random_results": [{"loss": 0.2, "p": 0.8, "s": 0.8}],
        }

        self.assertTrue(
            final_eval_json_matches_protocol(
                current,
                repeat_n=5,
                expect_random_groups=True,
            )
        )
        self.assertFalse(
            final_eval_json_matches_protocol(missing_random_repeat, repeat_n=5)
        )

    def test_compare_protocol_requires_random_groups_disabled(self):
        from rl_ga_compare_runner import final_eval_json_matches_protocol

        compare_ready = {
            "evaluation_protocol": {
                "version": 2,
                "baseline": "single_clean_validation_full",
                "noisy_repeat_n": 5,
                "random_groups": "disabled",
            },
            "baseline": {"loss": 0.1, "p": 0.9, "s": 0.9},
            "optimized": {"evaluation_n": 5},
            "random_results": [],
        }
        with_random = {
            **compare_ready,
            "evaluation_protocol": {
                **compare_ready["evaluation_protocol"],
                "random_groups": "enabled",
            },
            "random_results": [{"evaluation_n": 5}],
        }

        self.assertTrue(
            final_eval_json_matches_protocol(
                compare_ready,
                repeat_n=5,
                expect_random_groups=False,
            )
        )
        self.assertFalse(
            final_eval_json_matches_protocol(
                with_random,
                repeat_n=5,
                expect_random_groups=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
