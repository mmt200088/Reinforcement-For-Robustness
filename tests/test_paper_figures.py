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

            def guarded_read_jsonl_fields(path, **_kwargs):
                name = Path(path).name
                if name in {"episodes.jsonl", "ppo_updates.jsonl"}:
                    raise AssertionError(f"should not load {name} for cost_vs_accuracy-only")
                return []

            def guarded_read_json_file(path, **_kwargs):
                raise AssertionError(f"should not load JSON sidecar {Path(path).name}")

            def fake_cost_vs_accuracy(runs, **_kwargs):
                self.assertEqual(len(runs), 1)
                self.assertEqual(runs[0].episodes, [])
                self.assertEqual(runs[0].ppo_updates, [])
                return []

            with mock.patch.object(paper, "read_jsonl_fields", guarded_read_jsonl_fields):
                with mock.patch.object(paper, "read_json_file", guarded_read_json_file):
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

    def test_cost_vs_accuracy_streams_top_candidate_points_once_per_run(self):
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

        def fake_read_jsonl_xy(path, x_field, y_field):
            read_paths.append(path)
            self.assertEqual((x_field, y_field), ("total_bits", "total_reward"))
            return [10.0], [1.5]

        with mock.patch.object(paper, "_setup_matplotlib", return_value=FakePlt()):
            with mock.patch.object(paper, "_save_fig", return_value=[]):
                with mock.patch.object(
                    paper,
                    "read_jsonl_fields",
                    side_effect=AssertionError("cost_vs_accuracy should stream x/y points directly"),
                ):
                    with mock.patch.object(paper, "read_jsonl_xy", fake_read_jsonl_xy):
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

        self.assertEqual(run.episodes, [2.5])
        self.assertEqual(
            run.ppo_updates,
            [{"policy_loss": 0.1, "value_loss": 0.2, "entropy": 0.3, "clip_fraction": 0.4}],
        )

    def test_load_run_streams_episode_rewards_as_float_column(self):
        paper = _load_paper_figures_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            diag = root / "run" / "blb_stage2" / "progress" / "diagnostics"
            diag.mkdir(parents=True)
            (diag / "episodes.jsonl").write_text(
                json.dumps({"episode": 1, "total_reward": 2.5, "large_debug": "x" * 1000}) + "\n",
                encoding="utf-8",
            )
            seen = []

            def guarded_read_jsonl_fields(path, **kwargs):
                if Path(path).name == "episodes.jsonl":
                    raise AssertionError("episode rewards should stream as one float column")
                return []

            def fake_read_jsonl_float_field(path, field, **kwargs):
                seen.append((Path(path).name, field, kwargs))
                return [2.5]

            with mock.patch.object(paper, "read_jsonl_fields", guarded_read_jsonl_fields):
                with mock.patch.object(
                    paper,
                    "read_jsonl_float_field",
                    fake_read_jsonl_float_field,
                    create=True,
                ):
                    run = paper.load_run(
                        str(root / "run"),
                        include_ppo_updates=False,
                        include_best_action=False,
                        include_baseline_action=False,
                        include_first_invalid=False,
                        include_action_histogram=False,
                    )

        self.assertEqual(run.episodes, [2.5])
        self.assertEqual(seen, [("episodes.jsonl", "total_reward", {})])

    def test_load_run_reuses_json_native_action_payloads_without_copy(self):
        paper = _load_paper_figures_module()

        class NoCopyList(list):
            def __iter__(self):
                raise AssertionError("native JSON list payload should be reused, not copied")

        class NoCopyDict(dict):
            def keys(self):
                raise AssertionError("native JSON dict payload should be reused, not copied")

        action_vec = NoCopyList([1, 2, 3])
        best_slots = NoCopyList([{"slot": 1}])
        baseline_slots = NoCopyList([{"slot": 0}])
        diff_vs_baseline = NoCopyList([{"delta": -1}])
        first_invalid = NoCopyDict({"L00-B1": 2})

        def fake_read_json_file(path, **_kwargs):
            name = Path(path).name
            if name == "best_action_vec.json":
                return {
                    "action_vec": action_vec,
                    "slots": best_slots,
                    "diff_vs_baseline": diff_vs_baseline,
                }
            if name == "baseline_action_vec.json":
                return {"slots": baseline_slots}
            if name == "first_invalid_counts.json":
                return first_invalid
            return {}

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "run" / "blb_stage2" / "progress" / "diagnostics").mkdir(parents=True)
            with mock.patch.object(paper, "read_json_file", fake_read_json_file):
                run = paper.load_run(
                    str(root / "run"),
                    include_episodes=False,
                    include_ppo_updates=False,
                    include_action_histogram=False,
                )

        self.assertIs(run.best_action_vec, action_vec)
        self.assertIs(run.best_slots, best_slots)
        self.assertIs(run.baseline_slots, baseline_slots)
        self.assertIs(run.diff_vs_baseline, diff_vs_baseline)
        self.assertIs(run.first_invalid_counts, first_invalid)

    def test_training_curve_reuses_native_episode_rewards_without_copy(self):
        paper = _load_paper_figures_module()

        class NoCopyRewards(list):
            def __iter__(self):
                raise AssertionError("native episode reward list should be plotted without a copy")

        rewards = NoCopyRewards([1.0, 2.0, 3.0])
        run = paper.RunData(
            run_dir="/tmp/run",
            label="r1",
            progress_dir="/tmp/run/progress",
            episodes=rewards,
            ppo_updates=[],
            best_action_vec=[],
            best_slots=[],
            baseline_slots=[],
            diff_vs_baseline=[],
            first_invalid_counts={},
            action_histogram=None,
        )
        captured = {}

        class FakeAxes:
            def plot(self, _x, y, **_kwargs):
                captured["y"] = y

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

        with mock.patch.object(paper, "_setup_matplotlib", return_value=FakePlt()):
            with mock.patch.object(paper, "_save_fig", return_value=[]):
                paper.fig_training_curves([run], out_path_no_ext="/tmp/out", formats=("png",))

        self.assertIs(captured["y"], rewards)

    def test_group_training_curve_avoids_seed_slice_copies(self):
        paper = _load_paper_figures_module()

        class NoSliceRewards(list):
            def __getitem__(self, key):
                if isinstance(key, slice):
                    raise AssertionError("seed reward series should not be copied through slicing")
                return super().__getitem__(key)

        runs = [
            paper.RunData(
                run_dir=f"/tmp/run{i}",
                label=f"r{i}",
                progress_dir=f"/tmp/run{i}/progress",
                episodes=rewards,
                ppo_updates=[],
                best_action_vec=[],
                best_slots=[],
                baseline_slots=[],
                diff_vs_baseline=[],
                first_invalid_counts={},
                action_histogram=None,
            )
            for i, rewards in enumerate(
                [
                    NoSliceRewards([1.0, 2.0, 3.0, 4.0]),
                    NoSliceRewards([2.0, 4.0, 6.0]),
                ]
            )
        ]
        captured = {}

        class FakeAxes:
            def plot(self, _x, y, **_kwargs):
                captured["mean_len"] = len(y)

            def fill_between(self, _x, lower, upper, **_kwargs):
                captured["band_len"] = (len(lower), len(upper))

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

        with mock.patch.object(paper, "_setup_matplotlib", return_value=FakePlt()):
            with mock.patch.object(paper, "_save_fig", return_value=[]):
                paper.fig_training_curves(
                    runs,
                    group_label="seeded",
                    out_path_no_ext="/tmp/out",
                    formats=("png",),
                )

        self.assertEqual(captured["mean_len"], 3)
        self.assertEqual(captured["band_len"], (3, 3))

    def test_cli_reuses_static_figure_name_tuple_for_parser_defaults(self):
        source = PAPER_FIGURES_PATH.read_text(encoding="utf-8")
        cli_region = source[
            source.index("ALL_FIGS = {"):
            source.index("\n\nif __name__ == \"__main__\":")
        ]

        self.assertIn("ALL_FIG_NAMES = tuple(ALL_FIGS)", cli_region)
        self.assertIn("default=ALL_FIG_NAMES", cli_region)
        self.assertNotIn("list(ALL_FIGS.keys())", cli_region)


if __name__ == "__main__":
    unittest.main()
