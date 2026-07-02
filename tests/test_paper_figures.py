from __future__ import annotations

import importlib.util
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

        def fake_read_jsonl(path):
            read_paths.append(path)
            return [{"total_bits": 10, "total_reward": 1.5}]

        with mock.patch.object(paper, "_setup_matplotlib", return_value=FakePlt()):
            with mock.patch.object(paper, "_save_fig", return_value=[]):
                with mock.patch.object(paper, "_read_jsonl", fake_read_jsonl):
                    paper.fig_cost_vs_accuracy([run], out_path_no_ext="/tmp/out", formats=("png",))

        self.assertEqual(len(read_paths), 1)


if __name__ == "__main__":
    unittest.main()
