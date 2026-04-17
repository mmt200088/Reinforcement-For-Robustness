import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


try:
    import tools.status_board as status_board
except ImportError as exc:  # pragma: no cover
    status_board = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class StatusBoardTests(unittest.TestCase):
    def setUp(self):
        if status_board is None:
            self.skipTest(f"tools.status_board import unavailable: {_IMPORT_ERROR}")

    def test_rl_search_log_summary_uses_latest_incumbent(self):
        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "stage2_noise" / "pruning_search_log.txt"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                "\n".join(
                    [
                        "★ 守擂最优（incumbent best）已更新 · 回合 2821: 新 mean_score=0.6059, 成本 cost=41.45",
                        "★ 守擂最优（incumbent best）已更新 · 回合 61954: 新 mean_score=0.7076, 成本 cost=39.60",
                    ]
                ),
                encoding="utf-8",
            )

            summary = status_board._summarize_rl_search_log(log_path)

            self.assertEqual(summary, "S2 搜索 score=0.7076，cost=39.60，ep61954")

    def test_ga_search_log_summary_reads_generation_progress(self):
        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "stage2_noise" / "ga_noise_search_log.txt"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                "\n".join(
                    [
                        "[Stage2][Gen 2400/2500] best=1.300000(cost=38.00) incumbent=1.300000(cost=38.00)",
                        "[Stage2][Gen 2441/2500] best=1.415852(cost=36.90) incumbent=1.415852(cost=36.90)",
                    ]
                ),
                encoding="utf-8",
            )

            summary = status_board._summarize_ga_search_log(log_path)

            self.assertEqual(summary, "S2 搜索 score=1.4159，cost=36.90，gen2441/2500")

    def test_render_markdown_focuses_on_progress_and_best_result(self):
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "persistent_run"
            stage2_eval = run_dir / "stage2_noise_final_eval" / "noise_final_eval_results_mrpc.json"
            stage2_eval.parent.mkdir(parents=True, exist_ok=True)
            stage2_eval.write_text(
                json.dumps(
                    {
                        "dataset": "mrpc",
                        "selected": {
                            "gelu": [1] * 12,
                            "softmax": [2] * 12,
                            "noise_config": {
                                "input_noise_scaling_factors": [22] * 12,
                                "wq_noise_scaling_factors": [14] * 12,
                            },
                            "loss": 0.3268341392,
                            "p": 0.8784313725,
                            "s": 0.8763587246,
                            "tot_c": 37.75,
                            "tot_spd": 1.3033,
                            "feasible": True,
                            "evaluation_n": 5,
                            "loss_std": 0.0006230874,
                            "p_std": 0.0048029210,
                            "s_std": 0.0050463713,
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            rl_runs = [
                status_board.RunRecord(
                    algorithm="RL",
                    model_type="bert-base",
                    task="mrpc",
                    slug="s1t0.005_s2q0.05_s2sq0.05",
                    path=run_dir,
                    stage_status={
                        "stage1_search": "skipped",
                        "stage1_final_eval": "skipped",
                        "stage2_search": "completed",
                        "stage2_final_eval": "completed",
                    },
                )
            ]

            markdown = status_board.render_markdown(
                rl_runs=rl_runs,
                ga_runs=[],
                general_runs=[],
                compare_runs=[],
                experiments=[],
                root=Path(tmpdir),
                generated_at="2026-04-17 22:21:02",
            )

            self.assertIn("# 任务总板 / STATUS", markdown)
            self.assertIn("| 当前最优 | `S2 终评` · `loss=0.3268` · `主=0.8784` · `次=0.8764` · `cost=37.75` · `speed=1.30x` · `可行` |", markdown)
            self.assertIn("| 终评测试 | `n=5` · `loss=0.3268±0.0006` · `主=0.8784±0.0048` · `次=0.8764±0.0050` |", markdown)
            self.assertIn("**最优配置**", markdown)
            self.assertIn("[阶段1]", markdown)
            self.assertIn("gelu    [1, 1, 1, 1, 1, 1]", markdown)
            self.assertIn("        [1, 1, 1, 1, 1, 1]", markdown)
            self.assertIn("[阶段2]", markdown)
            self.assertRegex(markdown, r"x\s+\[22, 22, 22, 22, 22, 22\]")
            self.assertNotIn("生成时间", markdown)
            self.assertNotIn("最近更新", markdown)
            self.assertNotIn("| model | dataset | slug |", markdown)

    def test_single_layer_experiment_summary_reports_best_delta(self):
        with TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / "single_layer"
            exp_dir.mkdir(parents=True, exist_ok=True)
            (exp_dir / "single_layer_all_results.json").write_text(
                json.dumps(
                    [
                        {
                            "task": "sst2",
                            "primary_metric": "accuracy",
                            "baseline": {"accuracy": 0.9231},
                            "gelu_degradation": [{"accuracy": 0.9266}],
                            "softmax_degradation": [{"accuracy": 0.9243}],
                        },
                        {
                            "task": "mrpc",
                            "primary_metric": "accuracy",
                            "baseline": {"accuracy": 0.8823},
                            "gelu_degradation": [{"accuracy": 0.8830}],
                            "softmax_degradation": [{"accuracy": 0.8840}],
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            summary = status_board._summarize_single_layer(exp_dir)

            self.assertEqual(summary, "2 个任务；最佳 sst2 accuracy=0.9266（较 baseline +0.0035）")


if __name__ == "__main__":
    unittest.main()
