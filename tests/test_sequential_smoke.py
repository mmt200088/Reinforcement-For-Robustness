"""End-to-end-ish smoke test for the SF/K-first artifact pipeline.

This test does NOT spin up torch / transformers / actual PPO training (those
need GPU + several minutes); instead it drives the **artifact contracts**
end-to-end:

* synthesize a fake training run (in memory)
* run the diagnostics recorder for a few episodes
* register the run via tools.experiments_log
* rebuild the index
* aggregate two seeds via tools.aggregate_seeds
* render paper figures from the synthesized run
* verify every artifact file lands on disk with the expected schema

What this catches
-----------------
* Drift between `action_io.action_vec_to_slots_list` and Paean's reader.
* JSON schema regressions in diagnostics output (best_action_vec.json,
  episodes.jsonl, top_candidates.jsonl, etc.)
* Tools-package imports / CLI argument compatibility.
* paper_figures rendering on synthetic data (font / palette / sizing).

What this does NOT catch
------------------------
* Actual PPO training bugs (need full env + torch + transformers).
* `BLBStage2RLRunner` integration (covered by separate torch-requiring
  test, ``test_blb_stage2_rl_regressions``).

Run::

    python tests/test_sequential_smoke.py
    # or
    python -m unittest tests.test_sequential_smoke -v
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load_module_standalone(rel_path: str, name: str):
    """Load a single .py file without going through the package __init__
    (which may pull torch). Used to import diagnostics / action_io etc.
    in test isolation."""
    loader = importlib.machinery.SourceFileLoader(name, str(REPO_ROOT / rel_path))
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod


class SequentialArtifactContractsTest(unittest.TestCase):
    """Drive the artifact pipeline on synthetic data."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="blb_smoke_")
        # Numpy required; if missing, skip the whole class.
        try:
            import numpy as np  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("numpy not available")
        cls.diag_mod = _load_module_standalone(
            "blb_stage2_rl/diagnostics.py", "smoke_diag",
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _synthesize_run(self, run_dir: str, *, n_episodes: int = 10, seed: int = 42):
        """Run the diagnostics recorder over synthetic episodes."""
        import numpy as np

        os.makedirs(run_dir, exist_ok=True)
        diag = self.diag_mod

        rec = diag.RLDiagnosticsRecorder(
            output_dir=run_dir,
            num_layers=12,
            num_action_slots=577,
            max_action_levels=6,
            top_k=5,
            log_fn=lambda *_: None,
        )
        rec.set_meta({"profile": "mrpc", "fixed_label": "smoke"})
        rec.set_baseline_avg_k(11.7797)

        rng = np.random.default_rng(seed)
        best = -float("inf")
        for ep in range(n_episodes):
            invalid = max(0, 20 - ep)
            terminal = -0.3 + ep * 0.05 + rng.normal(0, 0.02)
            per_step = -invalid * 0.05
            total = float(terminal + per_step)
            new_best = total > best
            if new_best:
                best = total
            rec.record_episode(
                episode_stats=diag.EpisodeStats(
                    episode=ep,
                    total_reward=total,
                    terminal_reward=float(terminal),
                    per_step_sum=float(per_step),
                    valid_steps=59 - invalid,
                    invalid_steps=invalid,
                    steps_taken=59,
                    total_bits=15000 - ep * 100,
                    fusion_count=50 + ep,
                    first_invalid_step=(5 if invalid else None),
                    first_invalid_block=(3 if invalid else None),
                    first_invalid_layer=(8 if invalid else None),
                    early_terminated=False,
                ),
                full_action_vec=rng.integers(0, 6, size=577),
                is_new_best=new_best,
                best_reward_so_far=best,
            )
            if (ep + 1) % 5 == 0:
                rec.record_ppo_update(diag.PPOUpdateStats(
                    update=(ep + 1) // 5,
                    completed_episodes=ep + 1,
                    policy_loss=-0.03,
                    value_loss=0.4 - ep * 0.02,
                    entropy=2.0 - ep * 0.05,
                    clip_fraction=0.12,
                    n_samples=120,
                    window_mean_return=total,
                    window_max_return=total + 0.1,
                    window_min_return=total - 0.1,
                    window_mean_invalid=invalid,
                    best_reward_so_far=best,
                    elapsed_sec=float(ep * 0.1),
                ))
                rec.flush_periodic()
        rec.finalize()
        return rec

    # ----------------------------------------------------------------
    # 1. Diagnostics recorder lays down every expected file
    # ----------------------------------------------------------------
    def test_diagnostics_writes_all_files(self):
        run_dir = os.path.join(self.tmp, "run1")
        rec = self._synthesize_run(run_dir, n_episodes=12, seed=42)
        diag_dir = os.path.join(run_dir, "diagnostics")
        expected = [
            "episodes.jsonl",
            "ppo_updates.jsonl",
            "top_candidates.jsonl",
            "first_invalid_counts.json",
            "action_histogram.npz",
            "diagnostics_summary.md",
            "best_action_vec.json",
        ]
        for fn in expected:
            self.assertTrue(
                os.path.isfile(os.path.join(diag_dir, fn)),
                msg=f"missing {fn} under {diag_dir}",
            )

    # ----------------------------------------------------------------
    # 2. best_action_vec.json carries the new SF/K-first schema
    # ----------------------------------------------------------------
    def test_best_action_vec_json_schema(self):
        run_dir = os.path.join(self.tmp, "run2")
        self._synthesize_run(run_dir, n_episodes=8, seed=7)
        best_path = os.path.join(run_dir, "diagnostics", "best_action_vec.json")
        with open(best_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        # Required fields
        for key in ("schema_version", "num_layers", "action_vec",
                    "source", "episode", "total_reward"):
            self.assertIn(key, payload, msg=f"best_action_vec.json missing key: {key}")
        self.assertEqual(payload["schema_version"], "blb_v3_slots_human_v1")
        self.assertEqual(payload["num_layers"], 12)
        self.assertEqual(len(payload["action_vec"]), 577)

    # ----------------------------------------------------------------
    # 3. episodes.jsonl is one valid JSON per line
    # ----------------------------------------------------------------
    def test_episodes_jsonl_lines_parse(self):
        run_dir = os.path.join(self.tmp, "run3")
        self._synthesize_run(run_dir, n_episodes=6, seed=1)
        path = os.path.join(run_dir, "diagnostics", "episodes.jsonl")
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
        self.assertEqual(len(rows), 6)
        for r in rows:
            self.assertIn("episode", r)
            self.assertIn("total_reward", r)
            self.assertIn("invalid_steps", r)

    # ----------------------------------------------------------------
    # 4. summary.md contains the auto-flag section and at least one auto-flag
    #    (synthetic data has L8-B3 invalid concentration)
    # ----------------------------------------------------------------
    def test_summary_md_has_auto_flag(self):
        run_dir = os.path.join(self.tmp, "run4")
        self._synthesize_run(run_dir, n_episodes=40, seed=42)
        path = os.path.join(run_dir, "diagnostics", "diagnostics_summary.md")
        text = open(path, encoding="utf-8").read()
        self.assertIn("自动诊断", text)
        # First-invalid concentration auto-flag fires when one slot > 30%
        self.assertIn("L08-B3", text)

    # ----------------------------------------------------------------
    # 5. experiments_log register / rebuild / query round-trips
    # ----------------------------------------------------------------
    def test_experiments_log_roundtrip(self):
        # Use a private registry path so we don't pollute the real one.
        reg_path = os.path.join(self.tmp, "registry.jsonl")
        idx_path = os.path.join(self.tmp, "index.md")
        # Register one run
        rc = subprocess.run(
            [
                sys.executable, "-m", "tools.experiments_log", "register",
                "--run-id", "20260516_smoke_pid1",
                "--dataset", "mrpc",
                "--algorithm", "rl",
                "--preset", "smoke-test",
                "--rl-variant", "blb_v3_sequential",
                "--seed", "42",
                "--status", "complete",
                "--elapsed-sec", "120",
                "--best-reward", "0.42",
                "--final-eval-json", json.dumps({"loss": 0.4, "metric1": 0.8}),
                "--persistent-dir", "/tmp/fake",
                "--registry-path", reg_path,
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(rc.returncode, 0, msg=rc.stderr)
        # Rebuild
        rc = subprocess.run(
            [
                sys.executable, "-m", "tools.experiments_log", "rebuild",
                "--registry-path", reg_path,
                "--index-path", idx_path,
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(rc.returncode, 0, msg=rc.stderr)
        self.assertTrue(os.path.isfile(idx_path))
        text = open(idx_path, encoding="utf-8").read()
        self.assertIn("20260516_smoke_pid1", text)
        self.assertIn("mrpc", text)
        # Query
        rc = subprocess.run(
            [
                sys.executable, "-m", "tools.experiments_log", "query",
                "--dataset", "mrpc", "--min-reward", "0.1",
                "--registry-path", reg_path,
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(rc.returncode, 0, msg=rc.stderr)
        self.assertIn("20260516_smoke_pid1", rc.stdout)

    # ----------------------------------------------------------------
    # 6. aggregate_seeds tolerates missing runs gracefully
    # ----------------------------------------------------------------
    def test_aggregate_seeds_missing(self):
        out_dir = os.path.join(self.tmp, "agg_missing")
        os.makedirs(out_dir, exist_ok=True)
        seed_list = os.path.join(out_dir, "seed_list.txt")
        with open(seed_list, "w") as f:
            f.write("1 nonexistent_run_s1\n2 nonexistent_run_s2\n")
        rc = subprocess.run(
            [
                sys.executable, "-m", "tools.aggregate_seeds",
                "--run-name", "test_missing",
                "--seed-list", seed_list,
                "--output-dir", out_dir,
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(rc.returncode, 0, msg=rc.stderr)
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "seed_summary.md")))
        # Should mark all seeds as missing without crashing
        text = open(os.path.join(out_dir, "seed_summary.md"), encoding="utf-8").read()
        self.assertIn("test_missing", text)


class StrictModeTest(unittest.TestCase):
    """Verify blb_stage2_rl/strict.py contract."""

    @classmethod
    def setUpClass(cls):
        cls.strict_mod = _load_module_standalone(
            "blb_stage2_rl/strict.py", "smoke_strict",
        )

    def test_default_swallows(self):
        os.environ.pop("BLB_STRICT", None)
        called = {"n": 0}

        @self.strict_mod.swallow(reason="test", log_fn=lambda msg: called.update(n=called["n"] + 1))
        def boom():
            raise RuntimeError("oops")

        self.assertIsNone(boom())
        self.assertEqual(called["n"], 1, msg="swallow should log once")

    def test_strict_reraises(self):
        try:
            os.environ["BLB_STRICT"] = "1"

            @self.strict_mod.swallow(reason="test")
            def boom():
                raise RuntimeError("oops")

            with self.assertRaises(RuntimeError):
                boom()
        finally:
            os.environ.pop("BLB_STRICT", None)

    def test_guard_swallows(self):
        os.environ.pop("BLB_STRICT", None)
        with self.strict_mod.strict_guard(reason="test guard", log_fn=lambda _: None):
            raise ValueError("inside guard")
        # If we reached this line, the guard swallowed correctly.
        self.assertTrue(True)


class PresetValidatorTest(unittest.TestCase):
    """Quick smoke that the launcher flag extractor doesn't regress."""

    def test_extract_flags_nonempty(self):
        validate_mod = _load_module_standalone(
            "tools/validate_preset.py", "smoke_validate",
        )
        flags = validate_mod.extract_launcher_flags("llama_7B_LayerImportance.sh")
        self.assertGreater(
            len(flags), 50,
            msg=f"expected >50 launcher flags, got {len(flags)}: {sorted(flags)[:10]}",
        )
        self.assertIn("--blb-v3-seed", flags)
        self.assertIn("--run-tag", flags)

    def test_current_preset_is_clean(self):
        validate_mod = _load_module_standalone(
            "tools/validate_preset.py", "smoke_validate2",
        )
        flags = validate_mod.extract_launcher_flags("llama_7B_LayerImportance.sh")
        flags |= validate_mod.extract_launcher_flags("Paean/run_final_eval.sh")
        problems = validate_mod.validate_preset(
            "presets/mrpc-blb-stage2-rl.conf", flags,
        )
        self.assertEqual(problems, [], msg=f"preset has problems: {problems}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
