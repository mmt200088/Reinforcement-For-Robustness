"""Torch-free tests for the Stage-1-aligned Stage-2 RL outputs.

Covers the three pieces added to bring Stage-2 RL artifacts to parity with
Stage-1 RL:

  1. ``persistence.write_training_curves`` — Stage-1-style multi-panel curve
     (Reward / Loss / metric1 / metric2) + separate entropy
     curve, with optional series (back-compat with the legacy call).
  2. ``rl_local_optimum.{detect_rl_local_optimum, write_local_optimum_report}``
     — the local-optimum / health detection report (Stage-1 pruning_search_log
     format).
  3. ``scripts/blb_regen_stage2_outputs`` — the offline regenerator that reads a
     Stage-2 progress/ dir and re-emits the upgraded artifacts.
  4. ``config.run_layout.snapshot_decoupled_record`` for stage=2 with the exact
     final_config / final_eval / metadata shapes the sequential runner now
     builds (the record/ + COMPLETED archive that was previously dead code on
     the sequential path).

All torch-free: ``persistence`` and the regenerator are loaded via
``spec_from_file_location`` so the torch-importing ``blb_stage2_rl/__init__`` is
never triggered. matplotlib is optional (PNG asserts are skipped if absent);
the NPZ / text artifacts are always produced.
"""
import csv
import builtins
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import matplotlib  # noqa: F401
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


def _load_standalone(mod_name, rel_path):
    """Load a module by file path, bypassing package __init__ (torch-free)."""
    path = os.path.join(REPO_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


persistence = _load_standalone("blb_persistence_test", "blb_stage2_rl/persistence.py")
import rl_local_optimum as R  # noqa: E402  (numpy-only, torch-free)
from config import run_layout  # noqa: E402  (torch-free)


def _nonempty_file(path):
    return bool(path) and os.path.isfile(path) and os.path.getsize(path) > 0


class _CountingSequence:
    def __init__(self, values):
        self.values = list(values)
        self.iterations = 0

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        self.iterations += 1
        return iter(self.values)


class _IterOnlyText:
    def __init__(self, text):
        self._lines = text.splitlines(keepends=True)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self, *_args, **_kwargs):
        raise AssertionError("baseline parsing should scan lines instead of reading the whole file")

    def __iter__(self):
        return iter(self._lines)


class UpgradedCurvesTest(unittest.TestCase):
    def _full_kwargs(self, n=400):
        return dict(
            episode_returns=[float(i % 7 - 3) for i in range(n)],
            episode_losses=[0.3 + 0.001 * i for i in range(n)],
            episode_metric1s=[0.87 - 0.0002 * i for i in range(n)],
            episode_metric2s=[0.86 - 0.0002 * i for i in range(n)],
            episode_fusion_counts=[min(35, i // 10) for i in range(n)],
            episode_avg_ks=[13 - (i % 5) for i in range(n)],
            baselines={"loss": 0.30, "metric1": 0.87, "metric2": 0.86, "avg_k": 13.0},
            entropy_series=[1.0 - 0.001 * j for j in range(n // 4)],
            entropy_episodes=[4 * j for j in range(n // 4)],
        )

    def test_npz_always_written(self):
        with tempfile.TemporaryDirectory() as d:
            out = persistence.write_training_curves(d, **self._full_kwargs())
            self.assertTrue(_nonempty_file(out["npz"]),
                            "NPZ must always be written (matplotlib-independent)")

    def test_plot_rendering_defaults_to_offline_but_explicit_true_wins(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(persistence._stage2_plot_rendering_enabled(None))
            self.assertTrue(persistence._stage2_plot_rendering_enabled(True))
            self.assertFalse(persistence._stage2_plot_rendering_enabled(False))
        with mock.patch.dict(os.environ, {"RFR_STAGE2_RENDER_PLOTS": "1"}, clear=True):
            self.assertTrue(persistence._stage2_plot_rendering_enabled(None))

    def test_full_series_emits_all_pngs(self):
        if not HAVE_MPL:
            self.skipTest("matplotlib not installed")
        with tempfile.TemporaryDirectory() as d:
            out = persistence.write_training_curves(
                d, **self._full_kwargs(), render_plots=True,
            )
            for key in ("png", "entropy_png", "paper_png"):
                self.assertTrue(_nonempty_file(out[key]), f"{key} not produced")
            self.assertEqual(os.path.basename(out["png"]),
                             "blb_stage2_training_curve.png")
            self.assertEqual(os.path.basename(out["entropy_png"]),
                             "blb_stage2_entropy_curve.png")

    def test_render_plots_false_writes_npz_only(self):
        old = persistence.save_stage1_style_training_curve
        calls = []

        def fake_renderer(**_kwargs):
            calls.append(_kwargs)
            raise AssertionError("plot renderer should not be called")

        persistence.save_stage1_style_training_curve = fake_renderer
        try:
            with tempfile.TemporaryDirectory() as d:
                out = persistence.write_training_curves(
                    d,
                    **self._full_kwargs(),
                    render_plots=False,
                )
                self.assertTrue(_nonempty_file(out["npz"]))
                self.assertEqual(out["png"], "")
                self.assertEqual(out["entropy_png"], "")
                self.assertEqual(out["paper_png"], "")
                self.assertEqual(out["paper_pdf"], "")
                self.assertFalse(os.path.exists(os.path.join(d, "blb_stage2_training_curve.png")))
        finally:
            persistence.save_stage1_style_training_curve = old
        self.assertEqual(calls, [])

    def test_render_plots_false_iterates_each_npz_series_once(self):
        seqs = {
            "episode_returns": _CountingSequence([1.0, 2.0, 3.0]),
            "best_reward_curve": _CountingSequence([1.0, 2.0, 3.0]),
            "ppo_loss_curve": _CountingSequence([0.2, 0.1]),
            "episode_losses": _CountingSequence([0.3, 0.2, 0.1]),
            "episode_metric1s": _CountingSequence([0.8, 0.81, 0.82]),
            "episode_metric2s": _CountingSequence([0.7, 0.71, 0.72]),
            "episode_fusion_counts": _CountingSequence([0, 1, 2]),
            "episode_avg_ks": _CountingSequence([13, 12, 11]),
            "entropy_series": _CountingSequence([1.0, 0.9]),
            "entropy_episodes": _CountingSequence([120, 240]),
        }
        with tempfile.TemporaryDirectory() as d:
            out = persistence.write_training_curves(d, **seqs, render_plots=False)
            self.assertTrue(_nonempty_file(out["npz"]))
        for name, seq in seqs.items():
            self.assertLessEqual(seq.iterations, 1, name)

    def test_npz_writer_accepts_ndarray_without_list_materialization(self):
        import numpy as np

        values = np.asarray([1.0, 2.0, 3.0], dtype=float)
        captured = {}

        def fake_savez(path, **series):
            captured["series"] = series
            with open(path, "wb") as f:
                f.write(b"npz")

        with tempfile.TemporaryDirectory() as d:
            with mock.patch("numpy.savez", side_effect=fake_savez), \
                 mock.patch("builtins.list", side_effect=AssertionError("npz ndarray series should not be copied through list")):
                out = persistence.write_training_curves(
                    d,
                    episode_returns=values,
                    render_plots=False,
                )

            self.assertTrue(_nonempty_file(out["npz"]))

        written = captured["series"]["episode_returns"]
        self.assertIsInstance(written, np.ndarray)
        self.assertEqual(written.tolist(), [1.0, 2.0, 3.0])

    def test_curve_helpers_accept_ndarray_without_list_materialization(self):
        import numpy as np

        values = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=float)
        with mock.patch("builtins.list", side_effect=AssertionError("ndarray curve input should not be copied through list")):
            smoothed = persistence._ema_smooth(values, 3)
            ma_x, ma_y = persistence._moving_average(values, 2)

        self.assertEqual(smoothed.shape, values.shape)
        self.assertEqual(ma_x.tolist(), [2, 3, 4])
        self.assertEqual(ma_y.tolist(), [1.5, 3.0, 6.0])

    def test_float_array_accepts_tuple_without_list_materialization(self):
        values = (1.0, 2.0, 4.0)

        with mock.patch("builtins.list", side_effect=AssertionError("tuple curve input should not be copied through list")):
            arr = persistence._float_array(values)

        self.assertEqual(arr.tolist(), [1.0, 2.0, 4.0])

    def test_seq_len_counts_iterable_without_list_materialization(self):
        class IterOnly:
            def __iter__(self):
                return iter((1.0, 2.0, 3.0))

        with mock.patch("builtins.list", side_effect=AssertionError("iterable length should not be materialized through list")):
            self.assertEqual(persistence._seq_len(IterOnly()), 3)

    def test_stage1_style_panel_accepts_ndarray_without_list_materialization(self):
        import numpy as np

        class FakeAxis:
            def __init__(self):
                self.plots = 0

            def plot(self, *_args, **_kwargs):
                self.plots += 1

            def set_title(self, *_args, **_kwargs):
                pass

            def set_ylabel(self, *_args, **_kwargs):
                pass

            def set_xlabel(self, *_args, **_kwargs):
                pass

            def grid(self, *_args, **_kwargs):
                pass

            def legend(self, *_args, **_kwargs):
                pass

        axis = FakeAxis()
        raw = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=float)
        with mock.patch("builtins.list", side_effect=AssertionError("panel raw ndarray should not be copied through list")):
            persistence._stage1_style_panel(
                axis,
                raw,
                color="#000000",
                ma_color="#ffffff",
                ma_window=2,
                title="reward",
                ylabel="reward",
            )

        self.assertEqual(axis.plots, 2)

    def test_entropy_curve_accepts_ndarray_without_list_materialization(self):
        import numpy as np

        class FakeAxis:
            def __init__(self):
                self.plots = 0

            def plot(self, *_args, **_kwargs):
                self.plots += 1

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

        class FakeFigure:
            def tight_layout(self):
                pass

            def savefig(self, path, *_args, **_kwargs):
                with open(path, "wb") as f:
                    f.write(b"png")

        fake_axis = FakeAxis()
        fake_matplotlib = types.ModuleType("matplotlib")
        fake_matplotlib.use = lambda *_args, **_kwargs: None
        fake_pyplot = types.ModuleType("matplotlib.pyplot")
        fake_pyplot.subplots = lambda *_args, **_kwargs: (FakeFigure(), fake_axis)
        fake_pyplot.close = lambda *_args, **_kwargs: None
        fake_matplotlib.pyplot = fake_pyplot

        def fake_savez(path, **_series):
            with open(path, "wb") as f:
                f.write(b"npz")

        entropy = np.asarray([1.0, 0.9, 0.8, 0.7], dtype=float)
        entropy_episodes = np.asarray([120, 240, 360, 480], dtype=float)
        with tempfile.TemporaryDirectory() as d:
            with mock.patch("numpy.savez", side_effect=fake_savez), \
                 mock.patch.dict(sys.modules, {
                     "matplotlib": fake_matplotlib,
                     "matplotlib.pyplot": fake_pyplot,
                 }):
                with mock.patch("builtins.list", side_effect=AssertionError("entropy ndarray series should not be copied through list")):
                    out = persistence.write_training_curves(
                        d,
                        episode_returns=None,
                        entropy_series=entropy,
                        entropy_episodes=entropy_episodes,
                        render_plots=True,
                    )

            self.assertTrue(_nonempty_file(out["entropy_png"]))

        self.assertEqual(fake_axis.plots, 2)

    def test_env_can_disable_stage2_plot_rendering_without_callsite_change(self):
        old_env = os.environ.get("RFR_STAGE2_RENDER_PLOTS")
        os.environ["RFR_STAGE2_RENDER_PLOTS"] = "0"
        try:
            with tempfile.TemporaryDirectory() as d:
                out = persistence.write_training_curves(d, **self._full_kwargs())
                self.assertTrue(_nonempty_file(out["npz"]))
                self.assertEqual(out["png"], "")
                self.assertEqual(out["entropy_png"], "")
                self.assertEqual(out["paper_png"], "")
                self.assertEqual(out["paper_pdf"], "")
        finally:
            if old_env is None:
                os.environ.pop("RFR_STAGE2_RENDER_PLOTS", None)
            else:
                os.environ["RFR_STAGE2_RENDER_PLOTS"] = old_env

    def test_main_curve_uses_stage1_style_renderer_without_cost_panels(self):
        calls = []

        def fake_renderer(*, out_path, reward, loss, metric1, metric2,
                          baseline_loss=None, baseline_metric1=None,
                          baseline_metric2=None, metric1_name="metric1",
                          metric2_name=None, title_suffix="", moving_average_window=None):
            calls.append({
                "out_path": out_path,
                "reward": list(reward),
                "loss": list(loss),
                "metric1": list(metric1),
                "metric2": list(metric2) if metric2 is not None else None,
                "baseline_loss": baseline_loss,
                "baseline_metric1": baseline_metric1,
                "baseline_metric2": baseline_metric2,
                "metric1_name": metric1_name,
                "metric2_name": metric2_name,
                "title_suffix": title_suffix,
                "moving_average_window": moving_average_window,
            })
            with open(out_path, "wb") as f:
                f.write(b"fake png")
            return out_path

        old = persistence.save_stage1_style_training_curve
        persistence.save_stage1_style_training_curve = fake_renderer
        try:
            with tempfile.TemporaryDirectory() as d:
                out = persistence.write_training_curves(
                    d, **self._full_kwargs(n=40), render_plots=True,
                )
                self.assertTrue(_nonempty_file(out["png"]))
        finally:
            persistence.save_stage1_style_training_curve = old

        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(os.path.basename(call["out_path"]),
                         "blb_stage2_training_curve.png")
        self.assertEqual(len(call["reward"]), 40)
        self.assertEqual(len(call["loss"]), 40)
        self.assertEqual(len(call["metric1"]), 40)
        self.assertEqual(len(call["metric2"]), 40)
        self.assertEqual(call["baseline_loss"], 0.30)
        self.assertEqual(call["baseline_metric1"], 0.87)
        self.assertEqual(call["baseline_metric2"], 0.86)
        self.assertEqual(call["metric1_name"], "metric1")
        self.assertEqual(call["metric2_name"], "metric2")
        self.assertEqual(call["moving_average_window"], 24)

    def test_legacy_minimal_backcompat(self):
        # The old call (only returns + best + ppo_loss, no per-episode series)
        # must still work; entropy curve simply absent.
        with tempfile.TemporaryDirectory() as d:
            out = persistence.write_training_curves(
                d,
                episode_returns=[float(i) for i in range(50)],
                best_reward_curve=[49.0] * 50,
                ppo_loss_curve=[0.0] * 10,
                render_plots=True,
            )
            self.assertTrue(_nonempty_file(out["npz"]))
            self.assertEqual(out["entropy_png"], "",
                             "no entropy series → no entropy PNG")
            if HAVE_MPL:
                self.assertTrue(_nonempty_file(out["png"]))

    def test_no_baselines_ok(self):
        if not HAVE_MPL:
            self.skipTest("matplotlib not installed")
        kw = self._full_kwargs()
        kw.pop("baselines")
        with tempfile.TemporaryDirectory() as d:
            out = persistence.write_training_curves(d, **kw, render_plots=True)
            self.assertTrue(_nonempty_file(out["png"]))


class DetectionReportTest(unittest.TestCase):
    def test_report_has_all_signals_and_priority(self):
        ret = list(range(0, 100)) + [-7.0] * 900  # rises then frozen
        ent = [1.0] * 200 + [0.01] * 800
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "blb_stage2_search_log.txt")
            out = R.write_local_optimum_report(
                p, episode_returns=ret, episode_entropies=ent,
                completed_episodes=len(ret), title="BLB Stage-2 RL",
                extra_lines=["", "--- 优先级分布 ---", "  P1(acc):  5"],
            )
            self.assertTrue(_nonempty_file(out))
            text = open(out, encoding="utf-8").read()
            self.assertIn("=== BLB Stage-2 RL 局部最优检测报告 ===", text)
            for sig in ("entropy_collapsed", "reward_plateau", "best_stuck",
                        "action_diversity_collapsed"):
                self.assertIn(sig, text)
            self.assertIn("P1(acc)", text)

    def test_hot_collapse_flagged(self):
        ret = list(range(0, 300)) + [-6.95] * 1200  # frozen tail
        diag = R.detect_rl_local_optimum(ret, episode_entropies=[1.3] * 1500,
                                         window=300)
        self.assertTrue(diag["likely_local_optimum"])
        self.assertTrue(diag["signals"]["reward_plateau"])
        self.assertTrue(diag["signals"]["best_stuck"])

    def test_healthy_curve_ok(self):
        import numpy as np
        ret = list(np.linspace(0, 40, 1500) + np.random.RandomState(0).randn(1500))
        diag = R.detect_rl_local_optimum(ret, episode_entropies=[1.0] * 1500,
                                         window=300)
        self.assertFalse(diag["likely_local_optimum"])
        self.assertIn("[OK]", diag["summary"])


class StatusBoardLiveSummaryTest(unittest.TestCase):
    def test_status_board_labels_zero_episode_limit_as_unbounded(self):
        with tempfile.TemporaryDirectory() as d:
            board = persistence.BLBStatusBoard(
                d,
                total_episodes=0,
                profile="mrpc",
            )
            board.update_after_episode(120, 1.0, breakdown={"priority": 3})
            board.flush()
            with open(
                    os.path.join(d, "blb_stage2_live_summary.md"),
                    encoding="utf-8",
            ) as handle:
                text = handle.read()

        self.assertIn("Episode: 120 / unbounded", text)
        self.assertNotIn("Episode: 120 / 0", text)

    def test_status_board_flushes_human_readable_live_summary(self):
        with tempfile.TemporaryDirectory() as d:
            board = persistence.BLBStatusBoard(
                d,
                total_episodes=240,
                profile="mrpc",
                run_basename="s1t0.001_s2t0.001_s2st3.0",
            )
            board.set_phase("训练中")
            board.set_best(
                best_reward=1.25,
                best_action_vec=[1, 2, 3],
                best_breakdown={"priority": 3, "fusion_count": 36},
                best_episode=118,
            )
            board.update_after_episode(
                120,
                -30.0,
                breakdown={
                    "priority": 1,
                    "invalid": False,
                    "terminal_loss_mean": 0.34,
                    "terminal_metric1_mean": 0.88,
                    "terminal_metric2_mean": 0.87,
                    "fusion_count": 24,
                },
            )
            board.update_after_ppo_update(
                1,
                {
                    "policy_loss": 0.25,
                    "value_loss": 1.5,
                    "entropy": 2.0,
                    "clip_fraction": 0.1,
                    "window_mean_return": -12.0,
                    "window_mean_invalid": 0.0,
                },
            )

            summary_path = os.path.join(d, "blb_stage2_live_summary.md")
            self.assertTrue(_nonempty_file(summary_path))
            with open(summary_path, encoding="utf-8") as f:
                text = f.read()

        self.assertIn("# BLB Stage-2 RL Live Summary", text)
        self.assertIn("Episode: 120 / 240", text)
        self.assertIn("PPO updates: 1", text)
        self.assertIn("Best reward: +1.250000", text)
        self.assertIn("Last priority: 1", text)
        self.assertIn("policy_loss", text)
        self.assertIn("diagnostics/episodes.jsonl", text)


class EpisodeTraceMigrationTest(unittest.TestCase):
    def test_trace_append_skips_rechecking_known_current_schema(self):
        with tempfile.TemporaryDirectory() as d:
            persistence.append_blb_episode_trace_row(d, {"episode": 120})
            old_migrate = persistence._migrate_trace_schema_if_needed

            def fail_migrate(*_args, **_kwargs):
                raise AssertionError("current trace schema should be cached after first append")

            persistence._migrate_trace_schema_if_needed = fail_migrate
            try:
                trace_path = persistence.append_blb_episode_trace_row(d, {"episode": 240})
            finally:
                persistence._migrate_trace_schema_if_needed = old_migrate

            with open(trace_path, encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual([row["episode"] for row in rows], ["120", "240"])

    def test_trace_schema_migration_streams_rows_without_writerows(self):
        old_writerows = persistence.csv.DictWriter.writerows

        def fail_writerows(self, rowdicts):
            raise AssertionError("schema migration should stream rows")

        persistence.csv.DictWriter.writerows = fail_writerows
        try:
            with tempfile.TemporaryDirectory() as d:
                trace_path = os.path.join(d, persistence.BLB_EPISODE_TRACE_CSV)
                old_header = [
                    field for field in persistence.BLB_TRACE_FIELDNAMES
                    if field != "cost_probe_count"
                ]
                with open(trace_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(old_header)
                    for episode in (120, 240, 360):
                        row = {field: "" for field in old_header}
                        row.update({
                            "episode": episode,
                            "anchor_count": 1,
                            "policy_loss": 0.01,
                        })
                        writer.writerow([row.get(field, "") for field in old_header])

                persistence.append_blb_episode_trace_row(
                    d,
                    {
                        "episode": 480,
                        "anchor_count": 2,
                        "cost_probe_count": 3,
                        "policy_loss": 0.02,
                    },
                )
                with open(trace_path, encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))
        finally:
            persistence.csv.DictWriter.writerows = old_writerows

        self.assertEqual(list(rows[0].keys()), list(persistence.BLB_TRACE_FIELDNAMES))
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["cost_probe_count"], "0")
        self.assertEqual(rows[-1]["cost_probe_count"], "3")
        self.assertEqual(rows[-1]["episode"], "480")


class RegeneratorEndToEndTest(unittest.TestCase):
    def test_layerwise_action_table_decodes_bert_large_hml_actions(self):
        regen = _load_standalone(
            "blb_regen_layerwise_hml_table_test",
            "scripts/blb_regen_stage2_outputs.py",
        )
        action_matrix = [[0, 0], [1, 1], [1, 2]]
        action_matrix.extend([[0, 0] for _ in range(21)])

        rows = regen._layerwise_action_table(action_matrix)

        self.assertEqual(len(rows), 24)
        self.assertEqual(rows[0]["precision_preset"], "high")
        self.assertEqual(
            [rows[0][f"k_b{block_idx}"] for block_idx in range(1, 6)],
            [11, 10, 10, 12, 11],
        )
        self.assertEqual(rows[1]["block4_fusion"], 1)
        self.assertEqual(rows[1]["precision_preset"], "medium")
        self.assertEqual(
            [rows[1][f"k_b{block_idx}"] for block_idx in range(1, 6)],
            [9, 8, 8, 10, 9],
        )
        self.assertEqual(rows[2]["precision_preset"], "low")
        self.assertEqual(
            [rows[2][f"k_b{block_idx}"] for block_idx in range(1, 6)],
            [7, 6, 6, 8, 7],
        )

    def test_layerwise_action_table_decodes_all_twelve_layers_and_block3_k(self):
        regen = _load_standalone(
            "blb_regen_layerwise_table_test", "scripts/blb_regen_stage2_outputs.py"
        )
        action_matrix = [[0, 0, 2, 3, 4, 5]]
        action_matrix.extend([[1, 0, 1, 2, 3, 4] for _ in range(11)])

        rows = regen._layerwise_action_table(action_matrix)

        self.assertEqual(len(rows), 12)
        self.assertEqual(
            list(rows[0]),
            ["layer", "block4_fusion", "k_b1", "k_b2", "k_b3", "k_b4", "k_b5"],
        )
        self.assertEqual(rows[0]["k_b1"], 8)
        self.assertEqual(rows[0]["k_b3"], 13)
        self.assertEqual(rows[1]["block4_fusion"], 1)
        self.assertEqual(rows[1]["k_b3"], 11)

    def test_layerwise_html_report_shows_best_config_and_six_gates(self):
        regen = _load_standalone(
            "blb_regen_layerwise_html_test", "scripts/blb_regen_stage2_outputs.py"
        )
        matrix = [[0, 0, 3, 3, 3, 3]] + [[1, 3, 3, 3, 3, 3]] * 11
        summary = {
            "best_action_matrix": matrix,
            "best_variable_cost": 0.5,
            "best_resource_objective": {
                "compute_saving": 11 / 12,
                "communication_saving": 0.4,
                "robust_floor": 0.4,
                "secondary_progress": (11 / 12 + 0.4) / 2,
                "ppo_resource_score": 0.40002583075025826,
                "compute_shapley_credit": 0.2,
                "communication_shapley_credit": 0.20002583075025826,
            },
            "strict_pareto_frontier": [
                {
                    "candidate_key": "candidate-a",
                    "compute_saving": 11 / 12,
                    "communication_saving": 0.4,
                    "robust_floor": 0.4,
                    "secondary_progress": (11 / 12 + 0.4) / 2,
                }
            ],
            "best_metrics": {"loss_mean": 0.3, "metric1_mean": 0.88, "metric2_mean": 0.87},
            "best_assessment": {
                "loss_precision_probability": 0.96,
                "metric1_precision_probability": 0.97,
                "metric2_precision_probability": 0.98,
                "loss_stability_probability": 0.99,
                "metric1_stability_probability": 0.95,
                "metric2_stability_probability": 0.96,
            },
            "baseline_reference": {
                "groups": [
                    {
                        "group_index": 0,
                        "loss_trials": [0.30, 0.31, 0.32, 0.30, 0.31],
                        "metric1_trials": [0.88, 0.87, 0.89, 0.88, 0.88],
                        "metric2_trials": [0.87, 0.86, 0.88, 0.87, 0.87],
                    }
                ],
                "pooled": {
                    "trial_count": 25,
                    "loss_mean": 0.31,
                    "loss_std": 0.01,
                    "metric1_mean": 0.88,
                    "metric1_std": 0.01,
                    "metric2_mean": 0.87,
                    "metric2_std": 0.01,
                },
            },
            "best_promotion_evidence": {
                "status": "promoted",
                "trial_count": 25,
                "trials": {"seeds": list(range(25))},
            },
            "final_evidence": {
                "status": "pending_post_search_revalidation",
                "required_probability": 0.95,
                "required_trial_count": 25,
            },
        }
        with tempfile.TemporaryDirectory() as td:
            path = regen._write_layerwise_html_report(
                td,
                summary=summary,
                baseline={"loss_mean": 0.31, "metric1_mean": 0.88, "metric2_mean": 0.87},
                curve_paths={},
            )
            html = Path(path).read_text(encoding="utf-8")

        self.assertIn("Block4 Fusion", html)
        self.assertIn("K B3", html)
        self.assertIn("metric2_stability_probability", html)
        self.assertIn("0.500000", html)
        self.assertIn("Compute saving", html)
        self.assertIn("Communication saving", html)
        self.assertIn("Robust floor", html)
        self.assertIn("Compute Shapley credit", html)
        self.assertIn("Communication Shapley credit", html)
        self.assertIn("Strict Resource Pareto Frontier", html)
        self.assertIn("candidate-a", html)
        self.assertIn("Baseline Trial Distributions", html)
        self.assertIn("Promotion Evidence", html)
        self.assertIn("Final Revalidation Evidence", html)
        self.assertIn("pending_post_search_revalidation", html)
        self.assertNotIn("Policy Entropy by Action Type", html)

    def test_layerwise_baseline_parser_prefers_self_contained_summary(self):
        regen = _load_standalone(
            "blb_regen_layerwise_baseline_test", "scripts/blb_regen_stage2_outputs.py"
        )
        with tempfile.TemporaryDirectory() as td:
            progress = Path(td)
            (progress / "layerwise_summary.json").write_text(json.dumps({
                "baseline_reference": {
                    "pooled": {
                        "loss_mean": 0.31,
                        "loss_std": 0.011,
                        "metric1_mean": 0.88,
                        "metric1_std": 0.012,
                        "metric2_mean": 0.87,
                        "metric2_std": 0.013,
                    }
                }
            }), encoding="utf-8")

            parsed = regen._parse_baselines(str(progress))

        self.assertEqual(parsed["loss_mean"], 0.31)
        self.assertEqual(parsed["loss_std"], 0.011)
        self.assertEqual(parsed["metric1_std"], 0.012)
        self.assertEqual(parsed["metric2_std"], 0.013)

    def test_layerwise_search_report_uses_persisted_convergence_not_legacy_plateau(self):
        regen = _load_standalone(
            "blb_regen_layerwise_health_test", "scripts/blb_regen_stage2_outputs.py"
        )
        summary = {
            "schema_version": "stage2_layerwise_robust_summary_v1",
            "completed_episodes": 300,
            "converged": False,
            "block4_entropy": 0.4,
            "k_entropy": 0.5,
            "stall_update_windows": 2,
            "final_evidence": {"status": "pending_post_search_revalidation"},
        }
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(
                regen.rl_local_optimum,
                "write_local_optimum_report",
                side_effect=AssertionError("legacy plateau heuristic must stay disabled"),
            ):
                path = regen._write_search_health_report(
                    td,
                    persistence=types.SimpleNamespace(
                        BLB_SEARCH_LOG_TXT="blb_stage2_search_log.txt"
                    ),
                    layerwise_summary=summary,
                    episode_returns=[1.0] * 300,
                    entropies=[0.5],
                    priority=[3] * 300,
                    fusion_count=[24] * 300,
                    worst_signed_margin=None,
                    log_fn=lambda *_args: None,
                )
            text = Path(path).read_text(encoding="utf-8")

        self.assertIn("converged: False", text)
        self.assertIn("pending_post_search_revalidation", text)
        self.assertNotIn("local optimum", text.lower())

    def test_layerwise_html_report_renders_persisted_entropy_and_probability_curves(self):
        regen = _load_standalone(
            "blb_regen_layerwise_curve_test", "scripts/blb_regen_stage2_outputs.py"
        )
        probability_rows = [
            {
                "episode": episode,
                "fresh_constraint_probabilities": {
                    "loss_precision_probability": 0.50 + offset,
                    "metric1_precision_probability": 0.51 + offset,
                    "metric2_precision_probability": 0.52 + offset,
                    "loss_stability_probability": 0.53 + offset,
                    "metric1_stability_probability": 0.54 + offset,
                    "metric2_stability_probability": 0.55 + offset,
                },
                "pooled_constraint_probabilities": {
                    "loss_precision_probability": 0.70 + offset,
                    "metric1_precision_probability": 0.71 + offset,
                    "metric2_precision_probability": 0.72 + offset,
                    "loss_stability_probability": 0.73 + offset,
                    "metric1_stability_probability": 0.74 + offset,
                    "metric2_stability_probability": 0.75 + offset,
                },
            }
            for episode, offset in ((1, 0.0), (2, 0.1))
        ]
        update_rows = [
            {
                "completed_episodes": 120,
                "entropy": 1.5,
                "block4_entropy": 0.61,
                "k_entropy": 1.21,
            },
            {
                "completed_episodes": 240,
                "entropy": 1.4,
                "block4_entropy": 0.57,
                "k_entropy": 1.13,
            },
        ]
        matrix = [[0, 0, 3, 3, 3, 3]] + [[1, 3, 3, 3, 3, 3]] * 11
        with tempfile.TemporaryDirectory() as td:
            progress_dir = Path(td) / "progress"
            diagnostics_dir = progress_dir / "diagnostics"
            diagnostics_dir.mkdir(parents=True)
            (diagnostics_dir / "ppo_updates.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in update_rows),
                encoding="utf-8",
            )
            (diagnostics_dir / "episodes.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in probability_rows),
                encoding="utf-8",
            )

            curves = regen._read_layerwise_curves(str(progress_dir))
            self.assertEqual(
                curves["entropy"]["block4_entropy"],
                [(120.0, 0.61), (240.0, 0.57)],
            )
            self.assertEqual(
                curves["entropy"]["k_entropy"],
                [(120.0, 1.21), (240.0, 1.13)],
            )
            self.assertEqual(len(curves["fresh_constraint_probabilities"]), 6)
            self.assertEqual(len(curves["pooled_constraint_probabilities"]), 6)
            self.assertEqual(
                curves["fresh_constraint_probabilities"][
                    "loss_precision_probability"
                ][0][1],
                0.50,
            )
            self.assertEqual(
                curves["pooled_constraint_probabilities"][
                    "loss_precision_probability"
                ][0][1],
                0.70,
            )

            path = regen._write_layerwise_html_report(
                td,
                summary={"best_action_matrix": matrix},
                baseline={},
                curve_paths={},
                layerwise_curves=curves,
            )
            report_html = Path(path).read_text(encoding="utf-8")

        self.assertIn("Policy Entropy by Action Type", report_html)
        self.assertIn("Block4 fusion entropy", report_html)
        self.assertIn("Truncation K entropy", report_html)
        self.assertIn("Fresh Five-Trial Reward Constraint Probabilities", report_html)
        self.assertIn("Pooled Ranking and Promotion Constraint Probabilities", report_html)
        self.assertIn("loss_precision_probability", report_html)
        self.assertIn("metric2_stability_probability", report_html)
        self.assertGreaterEqual(report_html.count("<polyline"), 14)

    def test_layerwise_start_manifest_disables_legacy_heuristic_before_summary_exists(self):
        regen = _load_standalone(
            "blb_regen_layerwise_manifest_test", "scripts/blb_regen_stage2_outputs.py"
        )
        with tempfile.TemporaryDirectory() as td:
            progress = Path(td) / "progress"
            progress.mkdir()
            (progress / "layerwise_run_manifest.json").write_text(
                json.dumps({
                    "schema_version": "stage2_layerwise_robust_run_v1",
                    "status": "running",
                    "completed_episodes": 120,
                }),
                encoding="utf-8",
            )
            summary = regen._read_layerwise_summary(str(progress))
            with mock.patch.object(
                regen.rl_local_optimum,
                "write_local_optimum_report",
                side_effect=AssertionError("legacy heuristic must stay disabled"),
            ):
                report = regen._write_search_health_report(
                    td,
                    persistence=types.SimpleNamespace(
                        BLB_SEARCH_LOG_TXT="search.txt",
                    ),
                    layerwise_summary=summary,
                    episode_returns=[1.0],
                    entropies=[0.5],
                    priority=[3],
                    fusion_count=[24],
                    worst_signed_margin=None,
                    log_fn=lambda *_args: None,
                )
            report_text = Path(report).read_text(encoding="utf-8")

        self.assertEqual(summary["status"], "running")
        self.assertIn("layerwise robust ppo", report_text.lower())

    def test_layerwise_html_report_exists_without_a_strict_candidate(self):
        regen = _load_standalone(
            "blb_regen_layerwise_no_candidate_test",
            "scripts/blb_regen_stage2_outputs.py",
        )
        with tempfile.TemporaryDirectory() as td:
            report = regen._write_layerwise_html_report(
                td,
                summary={
                    "schema_version": "stage2_layerwise_robust_run_v1",
                    "status": "running",
                    "baseline_reference": {"groups": []},
                    "final_evidence": {"status": "no_candidate"},
                },
                baseline={"loss_mean": 0.3},
                curve_paths={},
                layerwise_curves={},
            )
            report_html = Path(report).read_text(encoding="utf-8")

        self.assertIn("No strict feasible candidate selected", report_html)
        self.assertIn("Final Revalidation Evidence", report_html)

    def _make_fake_run(self, d, gz=False):
        diag = os.path.join(d, "diagnostics")
        os.makedirs(diag, exist_ok=True)
        # episodes.jsonl (per-episode append form the recorder writes).
        rows = []
        for i in range(300):
            collapsed = i > 150
            fz = min(35, i // 5)
            rows.append({
                "episode": i,
                "per_step_sum": -2.0,
                "terminal_reward": (42.0 if not collapsed else -5.0),
                "terminal_loss_mean": (0.37 if not collapsed else 0.6),
                "terminal_metric1_mean": (0.87 if not collapsed else 0.70),
                "terminal_metric2_mean": (0.86 if not collapsed else 0.69),
                "fusion_count": fz,
                "terminal_k_gain": 2.0,
                "terminal_priority": (3 if not collapsed else 1),
                # ADR-014 debug fields (the regenerator should plot diagnostics
                # + emit collapse attribution from these).
                "fusion_count_b2": fz // 3,
                "fusion_count_b4": fz // 3,
                "fusion_count_b5": fz // 3,
                "terminal_worst_signed_margin": (0.6 - 0.05 * fz),
                "terminal_acc_barrier_sat": (-0.1 if not collapsed else 0.0),
                "terminal_acc_barrier_vio": (0.0 if not collapsed else -3.0),
                "terminal_cost_score": min(3.0, 0.1 * fz),
                "terminal_p3_metric_margin_reward": (0.3 if not collapsed else 0.0),
                "terminal_metric1_std": 0.0155,
            })
        ep_path = os.path.join(diag, "episodes.jsonl")
        if gz:
            import gzip
            with gzip.open(ep_path + ".gz", "wt", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        else:
            with open(ep_path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        # ppo_updates.jsonl (entropy source).
        with open(os.path.join(diag, "ppo_updates.jsonl"), "w", encoding="utf-8") as f:
            for u in range(1, 6):
                f.write(json.dumps({"update": u, "completed_episodes": u * 60,
                                    "entropy": 2.0 - 0.3 * u}) + "\n")
        # report.md §3 baseline table (baseline reference lines).
        with open(os.path.join(d, "blb_stage2_report.md"), "w", encoding="utf-8") as f:
            f.write("## 3. Baseline\n\n"
                    "| `avg_k` | 13.0 |\n"
                    "| `loss_mean` | 0.367 |\n"
                    "| `metric1_mean` | 0.871 |\n"
                    "| `metric2_mean` | 0.861 |\n")
        # status.json so _resolve_progress_dir finds it.
        with open(os.path.join(d, "blb_stage2_status.json"), "w", encoding="utf-8") as f:
            json.dump({"schema": "blb_stage2_status_v1"}, f)

    def _run(self, progress_dir, out_dir):
        regen = _load_standalone("blb_regen_test", "scripts/blb_regen_stage2_outputs.py")
        return regen.main([progress_dir, "--out-dir", out_dir,
                           "--metric1-name", "accuracy", "--metric2-name", "f1"])

    def test_episode_reader_does_not_materialize_absent_debug_fields(self):
        regen = _load_standalone("blb_regen_reader_test", "scripts/blb_regen_stage2_outputs.py")
        with tempfile.TemporaryDirectory() as d:
            diag = os.path.join(d, "diagnostics")
            os.makedirs(diag, exist_ok=True)
            with open(os.path.join(diag, "episodes.jsonl"), "w", encoding="utf-8") as f:
                for i in range(3):
                    f.write(json.dumps({
                        "per_step_sum": -1.0,
                        "terminal_reward": 2.0,
                        "terminal_loss_mean": 0.3,
                        "terminal_metric1_mean": 0.87,
                        "terminal_metric2_mean": 0.86,
                        "fusion_count": i,
                        "terminal_k_gain": 1.0,
                        "terminal_priority": 3,
                    }) + "\n")
            ep = regen._read_episodes(d)
        self.assertEqual(len(ep["returns"]), 3)
        self.assertEqual(ep["_present"], set())
        for key in (
            "fusion_b2", "fusion_b4", "fusion_b5", "worst_signed_margin",
            "acc_barrier_sat", "acc_barrier_vio", "cost_score",
            "p3_metric_margin", "metric1_std",
        ):
            self.assertEqual(ep[key], [], key)

    def test_jsonl_readers_use_shared_iter_jsonl(self):
        regen = _load_standalone("blb_regen_shared_jsonl_test", "scripts/blb_regen_stage2_outputs.py")
        calls = []

        episode_row = {
            "per_step_sum": -1.0,
            "terminal_reward": 2.0,
            "terminal_loss_mean": 0.3,
            "terminal_metric1_mean": 0.87,
            "terminal_metric2_mean": 0.86,
            "fusion_count": 1,
            "terminal_k_gain": 1.0,
            "terminal_priority": 3,
        }
        entropy_row = {"entropy": 1.5, "completed_episodes": 120}

        def fake_iter_jsonl(path, **kwargs):
            calls.append((os.path.basename(os.fspath(path)), kwargs))
            if os.path.basename(os.fspath(path)) == "episodes.jsonl":
                yield episode_row
            elif os.path.basename(os.fspath(path)) == "ppo_updates.jsonl":
                yield entropy_row
            else:
                raise AssertionError(path)

        with mock.patch.object(regen, "iter_jsonl", fake_iter_jsonl):
            ep = regen._read_episodes("unused")
            ent, ent_eps = regen._read_entropy("unused")

        self.assertEqual(ep["returns"], [1.0])
        self.assertEqual(ent, [1.5])
        self.assertEqual(ent_eps, [120.0])
        self.assertEqual(
            calls,
            [
                ("episodes.jsonl", {"gzip_fallback": True}),
                ("ppo_updates.jsonl", {"gzip_fallback": True}),
            ],
        )

    def test_baseline_parser_streams_report_and_summary(self):
        regen = _load_standalone("blb_regen_baseline_test", "scripts/blb_regen_stage2_outputs.py")
        with tempfile.TemporaryDirectory() as d:
            diag = os.path.join(d, "diagnostics")
            os.makedirs(diag, exist_ok=True)
            report = os.path.join(d, "blb_stage2_report.md")
            summary = os.path.join(diag, "diagnostics_summary.md")
            with open(report, "w", encoding="utf-8") as f:
                f.write(
                    "| `loss_mean` | 0.367 |\n"
                    "| `metric1_mean` | 0.871 |\n"
                    "| `metric2_mean` | 0.861 |\n"
                )
            with open(summary, "w", encoding="utf-8") as f:
                f.write("baseline avg_k: **13.0**\n")

            original_open = builtins.open

            def guarded_open(path, *args, **kwargs):
                if os.path.abspath(path) == os.path.abspath(report):
                    return _IterOnlyText(
                        "| `loss_mean` | 0.367 |\n"
                        "| `metric1_mean` | 0.871 |\n"
                        "| `metric2_mean` | 0.861 |\n"
                    )
                if os.path.abspath(path) == os.path.abspath(summary):
                    return _IterOnlyText("baseline avg_k: **13.0**\n")
                return original_open(path, *args, **kwargs)

            with mock.patch.object(builtins, "open", guarded_open):
                baselines = regen._parse_baselines(d)

        self.assertEqual(
            baselines,
            {
                "loss_mean": 0.367,
                "metric1_mean": 0.871,
                "metric2_mean": 0.861,
                "avg_k": 13.0,
            },
        )

    def test_regenerator_plain_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            self._make_fake_run(d, gz=False)
            out_dir = os.path.join(d, "preview")
            rc = self._run(d, out_dir)
            self.assertEqual(rc, 0)
            self.assertTrue(_nonempty_file(os.path.join(out_dir, "blb_stage2_search_log.txt")))
            self.assertTrue(_nonempty_file(os.path.join(out_dir, "blb_stage2_training_curve.npz")))
            if HAVE_MPL:
                self.assertTrue(_nonempty_file(os.path.join(out_dir, "blb_stage2_training_curve.png")))
                self.assertTrue(_nonempty_file(os.path.join(out_dir, "blb_stage2_entropy_curve.png")))
            # search log reflects the synthetic hot collapse + priority histogram
            # + ADR-014 collapse attribution (HOT verdict from the runaway fusion).
            with open(os.path.join(out_dir, "blb_stage2_search_log.txt"), encoding="utf-8") as f:
                text = f.read()
            self.assertIn("P3(cost)", text)
            self.assertIn("崩溃归因", text)
            self.assertIn("HOT", text)
            # diagnostics curve emitted from the ADR-014 debug fields.
            if HAVE_MPL:
                self.assertTrue(_nonempty_file(os.path.join(out_dir, "blb_stage2_diagnostics_curve.png")))

    def test_regenerator_gzip_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            self._make_fake_run(d, gz=True)
            out_dir = os.path.join(d, "preview")
            rc = self._run(d, out_dir)
            self.assertEqual(rc, 0)
            self.assertTrue(_nonempty_file(os.path.join(out_dir, "blb_stage2_training_curve.npz")))


class DecoupledArchiveShapeTest(unittest.TestCase):
    """The record/ + COMPLETED archive the sequential runner now produces.

    Exercises ``snapshot_decoupled_record(2, ...)`` with the exact
    final_config / final_eval / metadata shapes built in
    ``run_sequential_via_runner``'s §8.5 block.
    """

    def test_stage2_record_and_completed_marker(self):
        with tempfile.TemporaryDirectory() as root:
            combo = "bert base mrpc"
            wd = os.path.join(root, "stage2", combo)
            os.makedirs(wd, exist_ok=True)
            final_config = {
                "stage": 2, "combo": combo, "profile": "mrpc", "num_layers": 12,
                "blb_v3_best_action_vec": [14, 0, 3],
                "gelu_degree_per_layer": [1] * 12,
                "softmax_degree_per_layer": [6] * 12,
            }
            final_eval = {
                "source": "training_best_mean_of_K_trials",
                "best_reward": 40.8,
                "loss": 0.37, "metric1": 0.86, "metric2": 0.86,
                "cost": {"total_bits_sum": 10575, "total_fusion_count": 22, "avg_k": 11.0},
                "baseline_cost": {"total_bits_sum": 11285, "total_fusion_count": 0,
                                  "avg_k": 13.0, "loss_mean": 0.367, "metric1_mean": 0.871},
            }
            metadata = {"stage": 2, "combo": combo, "profile": "mrpc",
                        "stage2_limit_tolerance": 0.005, "stage2_stability_tolerance": 5.0}
            rdir, rid, n = run_layout.snapshot_decoupled_record(
                2, combo, wd,
                final_config=final_config, final_eval=final_eval, metadata=metadata,
                curve_paths=[], report_md="# Stage-2 record: bert base mrpc\n",
                root=root,
            )
            # record/ has the four canonical files + report.md
            for name in ("final_config.json", "final_eval.json", "metadata.json", "report.md"):
                self.assertTrue(_nonempty_file(os.path.join(rdir, name)), f"missing {name}")
            # COMPLETED marker in working dir
            self.assertTrue(run_layout.is_completed(wd))
            # round-trip a field
            with open(os.path.join(rdir, "final_config.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f)["stage"], 2)
            self.assertEqual(n, 1)


class DiagnosticCurvesTest(unittest.TestCase):
    """ADR-014 ``write_diagnostic_curves`` (collapse diagnostics PNG)."""

    def test_diagnostic_curves_accept_tuple_series_without_list_materialization(self):
        class FakeAxis:
            def __init__(self):
                self.plots = 0

            def plot(self, *_args, **_kwargs):
                self.plots += 1

            def set_ylim(self, *_args, **_kwargs):
                pass

            def set_ylabel(self, *_args, **_kwargs):
                pass

            def axhline(self, *_args, **_kwargs):
                pass

            def set_xlabel(self, *_args, **_kwargs):
                pass

            def set_title(self, *_args, **_kwargs):
                pass

            def grid(self, *_args, **_kwargs):
                pass

            def legend(self, *_args, **_kwargs):
                pass

        class FakeAxes:
            def __init__(self, axes):
                self._axes = axes

            def __getitem__(self, item):
                row, col = item
                if col != 0:
                    raise AssertionError(f"unexpected column index {col}")
                return self._axes[row]

        class FakeFigure:
            def suptitle(self, *_args, **_kwargs):
                pass

            def tight_layout(self, *_args, **_kwargs):
                pass

            def savefig(self, path, *_args, **_kwargs):
                with open(path, "wb") as f:
                    f.write(b"png")

        fake_axes = [FakeAxis() for _ in range(2)]
        fake_matplotlib = types.ModuleType("matplotlib")
        fake_matplotlib.use = lambda *_args, **_kwargs: None
        fake_pyplot = types.ModuleType("matplotlib.pyplot")
        fake_pyplot.subplots = lambda *_args, **_kwargs: (FakeFigure(), FakeAxes(fake_axes))
        fake_pyplot.close = lambda *_args, **_kwargs: None
        fake_matplotlib.pyplot = fake_pyplot

        priority = (3, 3, 2, 1)
        fusion = (0.0, 1.0, 2.0, 3.0)
        logs = []
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(sys.modules, {
                    "matplotlib": fake_matplotlib,
                    "matplotlib.pyplot": fake_pyplot,
            }):
                with mock.patch("builtins.list", side_effect=AssertionError("diagnostic tuple series should not be copied through list")):
                    out = persistence.write_diagnostic_curves(
                        d,
                        priority=priority,
                        fusion_count=fusion,
                        rolling_window=2,
                        log_fn=logs.append,
                        render_plots=True,
                    )

            if not _nonempty_file(out["diagnostics_png"]):
                self.fail("diagnostic render failed: " + "\n".join(logs))

        self.assertGreater(sum(axis.plots for axis in fake_axes), 0)

    def test_emits_png_with_full_series(self):
        if not HAVE_MPL:
            self.skipTest("matplotlib not installed")
        n = 400
        fz = [min(35, i // 10) for i in range(n)]
        with tempfile.TemporaryDirectory() as d:
            out = persistence.write_diagnostic_curves(
                d,
                priority=[3 if f < 12 else 1 for f in fz],
                fusion_count=fz,
                fusion_b2=[f / 3 for f in fz], fusion_b4=[f / 3 for f in fz],
                fusion_b5=[f / 3 for f in fz],
                worst_signed_margin=[0.6 - 0.05 * f for f in fz],
                acc_barrier_sat=[-0.1] * n, acc_barrier_vio=[0.0] * n,
                cost_score=[min(3.0, 0.1 * f) for f in fz],
                p3_metric_margin=[0.3] * n, metric1_std=[0.0155] * n,
                rolling_window=100,
                render_plots=True,
            )
            self.assertTrue(_nonempty_file(out["diagnostics_png"]))

    def test_render_plots_false_skips_diagnostic_png(self):
        with tempfile.TemporaryDirectory() as d:
            out = persistence.write_diagnostic_curves(
                d,
                priority=[1, 2, 3],
                fusion_count=[0, 1, 2],
                render_plots=False,
            )
            self.assertEqual(out["diagnostics_png"], "")
            self.assertFalse(os.path.exists(os.path.join(d, "blb_stage2_diagnostics_curve.png")))

    def test_no_data_is_safe(self):
        with tempfile.TemporaryDirectory() as d:
            out = persistence.write_diagnostic_curves(d, priority=None, fusion_count=None)
            self.assertEqual(out["diagnostics_png"], "")


class PersistedDebugFieldsTest(unittest.TestCase):
    """ADR-014 B1: barrier/margin + per-block fusion reach episodes.jsonl, and
    the rolling-health log is written (the previously black-box mechanism)."""

    def test_episode_stats_round_trip_and_health_log(self):
        diag = _load_standalone("blb_diag_test", "blb_stage2_rl/diagnostics.py")
        with tempfile.TemporaryDirectory() as d:
            rec = diag.RLDiagnosticsRecorder(
                output_dir=d, num_layers=12, num_action_slots=47, max_action_levels=6)
            for ep in range(4):
                st = diag.EpisodeStats(
                    episode=ep, total_reward=40 - ep, terminal_reward=42 - ep,
                    per_step_sum=-2.0, valid_steps=47, invalid_steps=0, steps_taken=47,
                    total_bits=10000, fusion_count=ep * 4,
                    first_invalid_step=None, first_invalid_block=None,
                    first_invalid_layer=None, early_terminated=False,
                    fusion_count_b2=ep, fusion_count_b4=ep, fusion_count_b5=ep,
                    terminal_final_config_fingerprint=f"cfg-{ep}",
                    terminal_materialization_failure_reason="",
                    terminal_model_uses_replan_config=True,
                    terminal_priority=(3 if ep < 2 else 1),
                    terminal_worst_signed_margin=0.5 - 0.2 * ep,
                    terminal_acc_barrier_sat=-0.1 * ep, terminal_acc_barrier_vio=0.0,
                    terminal_near_miss=False, terminal_margin_m1=0.5 - 0.2 * ep,
                    terminal_margin_m2=0.6 - 0.2 * ep,
                    terminal_fusion_norm_raw=ep / 11.0,
                    terminal_fusion_norm_saturated=min(1.0, (ep / 11.0) / 0.15))
                rec.record_episode(episode_stats=st, full_action_vec=None,
                                   is_new_best=(ep == 0), best_reward_so_far=40.0)
            rec.flush_periodic()
            last = open(os.path.join(d, "diagnostics", "episodes.jsonl"),
                        encoding="utf-8").read().strip().splitlines()[-1]
            j = json.loads(last)
            for k in ("terminal_worst_signed_margin", "terminal_acc_barrier_sat",
                      "terminal_acc_barrier_vio", "terminal_near_miss",
                      "terminal_margin_m1", "terminal_margin_m2",
                      "terminal_fusion_norm_raw", "terminal_fusion_norm_saturated",
                      "terminal_final_config_fingerprint",
                      "terminal_materialization_failure_reason",
                      "terminal_model_uses_replan_config"):
                self.assertIn(k, j, f"episodes.jsonl missing {k}")
            self.assertEqual(j["terminal_final_config_fingerprint"], "cfg-3")
            self.assertTrue(j["terminal_model_uses_replan_config"])
            # rolling-health log written with the expected columns.
            hp = os.path.join(d, "diagnostics", "blb_stage2_health.log")
            self.assertTrue(_nonempty_file(hp))
            line = open(hp, encoding="utf-8").read().strip()
            for token in ("rolling", "P1=", "P3=", "fusion=", "margin="):
                self.assertIn(token, line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
