"""Torch-free tests for the Stage-1-aligned Stage-2 RL outputs.

Covers the three pieces added to bring Stage-2 RL artifacts to parity with
Stage-1 RL:

  1. ``persistence.write_training_curves`` — Stage-1-style multi-panel curve
     (Reward / Loss / metric1 / metric2 / fusion / avg_K) + separate entropy
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
import importlib.util
import json
import os
import sys
import tempfile
import unittest

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

    def test_full_series_emits_all_pngs(self):
        if not HAVE_MPL:
            self.skipTest("matplotlib not installed")
        with tempfile.TemporaryDirectory() as d:
            out = persistence.write_training_curves(d, **self._full_kwargs())
            for key in ("png", "entropy_png", "paper_png"):
                self.assertTrue(_nonempty_file(out[key]), f"{key} not produced")
            self.assertEqual(os.path.basename(out["png"]),
                             "blb_stage2_training_curve.png")
            self.assertEqual(os.path.basename(out["entropy_png"]),
                             "blb_stage2_entropy_curve.png")

    def test_legacy_minimal_backcompat(self):
        # The old call (only returns + best + ppo_loss, no per-episode series)
        # must still work; entropy curve simply absent.
        with tempfile.TemporaryDirectory() as d:
            out = persistence.write_training_curves(
                d,
                episode_returns=[float(i) for i in range(50)],
                best_reward_curve=[49.0] * 50,
                ppo_loss_curve=[0.0] * 10,
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
            out = persistence.write_training_curves(d, **kw)
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


class RegeneratorEndToEndTest(unittest.TestCase):
    def _make_fake_run(self, d, gz=False):
        diag = os.path.join(d, "diagnostics")
        os.makedirs(diag, exist_ok=True)
        # episodes.jsonl (per-episode append form the recorder writes).
        rows = []
        for i in range(300):
            collapsed = i > 150
            rows.append({
                "episode": i,
                "per_step_sum": -2.0,
                "terminal_reward": (42.0 if not collapsed else -5.0),
                "terminal_loss_mean": (0.37 if not collapsed else 0.6),
                "terminal_metric1_mean": (0.87 if not collapsed else 0.70),
                "terminal_metric2_mean": (0.86 if not collapsed else 0.69),
                "fusion_count": min(35, i // 5),
                "terminal_k_gain": 2.0,
                "terminal_priority": (3 if not collapsed else 1),
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
            # search log reflects the synthetic hot collapse + priority histogram.
            text = open(os.path.join(out_dir, "blb_stage2_search_log.txt"), encoding="utf-8").read()
            self.assertIn("P3(cost)", text)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
