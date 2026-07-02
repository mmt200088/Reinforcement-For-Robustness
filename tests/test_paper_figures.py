from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_FIGURES_PATH = REPO_ROOT / "tools" / "paper_figures.py"


def _load_paper_figures_module():
    spec = importlib.util.spec_from_file_location("paper_figures", PAPER_FIGURES_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules["paper_figures"] = module
    spec.loader.exec_module(module)
    return module


class PaperFiguresTest(unittest.TestCase):
    def test_cost_vs_accuracy_only_skips_unneeded_large_run_artifacts(self):
        paper = _load_paper_figures_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "run"
            diag = run_dir / "blb_stage2" / "progress" / "diagnostics"
            diag.mkdir(parents=True)
            out_dir = root / "figures"

            def guarded_read_jsonl(path):
                name = Path(path).name
                if name in {"episodes.jsonl", "ppo_updates.jsonl"}:
                    raise AssertionError(f"should not load {name} for cost_vs_accuracy-only")
                return []

            def guarded_read_json(path):
                raise AssertionError(f"should not load JSON sidecar {Path(path).name}")

            def fake_cost_vs_accuracy(runs, **_kwargs):
                self.assertEqual(len(runs), 1)
                self.assertEqual(runs[0].episodes, [])
                self.assertEqual(runs[0].ppo_updates, [])
                return []

            with mock.patch.object(paper, "_read_jsonl", guarded_read_jsonl):
                with mock.patch.object(paper, "_read_json", guarded_read_json):
                    with mock.patch.object(paper, "fig_cost_vs_accuracy", fake_cost_vs_accuracy):
                        rc = paper.main([
                            "--runs",
                            str(run_dir),
                            "--out",
                            str(out_dir),
                            "--figs",
                            "cost_vs_accuracy",
                            "--formats",
                            "png",
                        ])

        self.assertEqual(rc, 0)

    def test_cost_vs_accuracy_reads_top_candidates_once_per_run(self):
        paper = _load_paper_figures_module()
        run = paper.RunData(
            run_dir="/tmp/run",
            label="r1",
            progress_dir="/tmp/run/progress",
            episodes=[],
            ppo_updates=[],
            best_action_vec=[],
            best_slots=[],
            baseline_slots=[],
            diff_vs_baseline=[],
            first_invalid_counts={},
            action_histogram=None,
        )
        read_paths = []

        class FakeAxes:
            def scatter(self, *_args, **_kwargs):
                pass

            def set_xlabel(self, *_args, **_kwargs):
                pass

            def set_ylabel(self, *_args, **_kwargs):
                pass

            def set_title(self, *_args, **_kwargs):
                pass

            def grid(self, *_args, **_kwargs):
                pass

            def legend(self, *_args, **_kwargs):
                pass

        class FakePlt:
            def subplots(self, **_kwargs):
                return object(), FakeAxes()

        def fake_read_jsonl(path, fields=None):
            read_paths.append(path)
            self.assertEqual(tuple(fields or ()), ("total_bits", "total_reward"))
            return [{"total_bits": 10, "total_reward": 1.5}]

        with mock.patch.object(paper, "_setup_matplotlib", return_value=FakePlt()):
            with mock.patch.object(paper, "_save_fig", return_value=[]):
                with mock.patch.object(paper, "_read_jsonl", fake_read_jsonl):
                    paper.fig_cost_vs_accuracy([run], out_path_no_ext="/tmp/out", formats=("png",))

        self.assertEqual(len(read_paths), 1)

    def test_load_run_projects_large_jsonl_rows_to_needed_fields(self):
        paper = _load_paper_figures_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            diag = root / "run" / "blb_stage2" / "progress" / "diagnostics"
            diag.mkdir(parents=True)
            (diag / "episodes.jsonl").write_text(
                json.dumps({"episode": 1, "total_reward": 2.5, "large_debug": "x" * 1000}) + "\n",
                encoding="utf-8",
            )
            (diag / "ppo_updates.jsonl").write_text(
                json.dumps(
                    {
                        "policy_loss": 0.1,
                        "value_loss": 0.2,
                        "entropy": 0.3,
                        "clip_fraction": 0.4,
                        "large_debug": "y" * 1000,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            run = paper.load_run(
                str(root / "run"),
                include_best_action=False,
                include_baseline_action=False,
                include_first_invalid=False,
                include_action_histogram=False,
            )

        self.assertEqual(run.episodes, [{"total_reward": 2.5}])
        self.assertEqual(
            run.ppo_updates,
            [{"policy_loss": 0.1, "value_loss": 0.2, "entropy": 0.3, "clip_fraction": 0.4}],
        )


if __name__ == "__main__":
    unittest.main()
