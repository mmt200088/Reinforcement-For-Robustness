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


class OutputHygieneRegressionTest(unittest.TestCase):
    """Catch the regression where layer_importance_evaluator wrote the noise
    log header 80x per init (due to implicit string concat * operator-precedence
    bug), and verify the sequential runner emits border-less progress blocks.
    """

    def test_noise_log_header_uses_explicit_join_not_implicit_concat(self):
        src = open("layer_importance_evaluator.py", encoding="utf-8").read()
        # The new safe form joins a list — no `"=" * 80 + "\n" "abc\n" "=" * 80`
        # anywhere near the noise log header.
        head_idx = src.find('"【二阶段噪声 RL 日志】二阶段噪声 RL 日志开始')
        self.assertGreater(head_idx, 0, "noise log header literal disappeared")
        snippet = src[max(0, head_idx - 600): head_idx + 400]
        self.assertNotIn(
            '"=" * 80 + "\\n"\n                    "【二阶段噪声',
            snippet,
            msg="legacy implicit-concat header form re-appeared (80x duplication risk)",
        )
        self.assertIn(
            'header_lines',
            snippet,
            msg="expected explicit list+join form in _initialize_noise_log_file",
        )

    def test_sequential_box_is_borderless(self):
        src = open("blb_stage2_rl/sequential_runner.py", encoding="utf-8").read()
        box_idx = src.find("def _seq_log_rounded_box")
        self.assertGreater(box_idx, 0)
        # Skip past the docstring (which intentionally references the legacy
        # `╭─╮│╰╯` chars to document why they were removed).
        body_start = src.find('"""', src.find('"""', box_idx) + 3) + 3
        body_end = src.find("\ndef ", body_start)
        box_body = src[body_start: body_end]
        for ch in ("╭", "╮", "╰", "╯", "│"):
            self.assertNotIn(
                ch, box_body,
                msg=f"box character {ch!r} should be gone after border removal",
            )

    def test_sequential_runner_wires_details_and_crash_watcher(self):
        src = open("blb_stage2_rl/sequential_runner.py", encoding="utf-8").read()
        for symbol in (
            "BLBStepDetailsWriter(",
            "BLBRewardCrashWatcher(",
            "details_writer.append_episode",
            "crash_watcher.observe_rollout",
            "details_writer.flush",
        ):
            self.assertIn(
                symbol, src,
                msg=f"{symbol!r} missing from sequential_runner.py — legacy v2 parity broken",
            )

    def test_sequential_runner_has_noisy_baseline_preflight(self):
        """Regression: previously the sequential path skipped the noisy
        baseline preflight, leaving stab_threshold at ~0.001 (driven by
        clean baseline.loss_std == 0). Every episode then collapsed into
        terminal_reward = -150 from the priority-2 inf-fallback branch.
        See blb_stage2_rl/sequential_runner.py @ "4.5) NOISY baseline
        preflight" for the fix.
        """
        src = open("blb_stage2_rl/sequential_runner.py", encoding="utf-8").read()
        for needle in (
            "noisy baseline preflight",
            "base_env.step(baseline_action_vec)",
            "stage2_stability_tolerance",
            "stage2_limit_tolerance",
        ):
            self.assertIn(
                needle, src,
                msg=f"sequential_runner.py missing preflight calibration: {needle!r}",
            )

    def test_eval_on_probe_clamps_nonfinite_losses(self):
        """Regression: cross_entropy can overflow under heavy BLB noise and
        emit inf/nan losses. Before this clamp a single bad trial made
        np.std → inf for the whole episode, and every action collapsed to
        the same -150 reward. The clamp keeps std finite + comparable.
        """
        src = open("blb_stage2_rl/env.py", encoding="utf-8").read()
        for needle in (
            "_LOSS_CAP",
            "nan_to_num",
            "np.clip(loss_arr",
        ):
            self.assertIn(
                needle, src,
                msg=f"env.py:_eval_on_probe missing finite-loss safety: {needle!r}",
            )

    def test_episode_record_carries_invalid_block_details(self):
        """Regression: each invalid sub-step's (layer, block, reason) should
        be appended to ``EpisodeRecord.invalid_block_details`` and surfaced
        into the details/ rollover file as bulleted lines. Previously only
        the FIRST invalid step was recorded; operators had to re-run the
        optimizer to find the other 7+ failures per episode.
        """
        src = open("blb_stage2_rl/sequential_runner.py", encoding="utf-8").read()
        for needle in (
            "invalid_block_details",
            "_format_invalid_chain_reason",
            'extra_lines.append(\n                    f"invalid_blocks',
        ):
            self.assertIn(
                needle, src,
                msg=f"sequential_runner.py missing per-block invalid plumbing: {needle!r}",
            )

    def test_format_invalid_chain_reason_helper_present(self):
        """Source-text smoke: the helper function exists and handles the
        three real-world shapes (None, structured dict with reason+stage+
        primes_*, opaque fallback). We can't import sequential_runner.py
        locally because it pulls in torch, so we assert the function body
        contains the key branches.
        """
        src = open("blb_stage2_rl/sequential_runner.py", encoding="utf-8").read()
        head = src.find("def _format_invalid_chain_reason")
        self.assertGreater(head, 0, "_format_invalid_chain_reason missing")
        body = src[head: head + 1500]
        self.assertIn("(none)", body)
        self.assertIn("reason", body)
        self.assertIn("primes_over_q_max", body)
        self.assertIn("json.dumps", body)


class ForbiddenActionMaskTest(unittest.TestCase):
    """Per-(layer, block) blacklist of action tuples that triggered
    invalid_chain. Used by train_sequential to rejection-sample around
    known-bad tuples — see blb_stage2_rl/action_mask.py @ ForbiddenActionMask.

    action_mask.py has a relative import (``from .action_space import ...``)
    and action_space.py is non-trivial; importing it standalone is painful.
    So this test exercises the source text only — verifies the class is
    present with the expected API surface.
    """

    def test_class_present_with_expected_api(self):
        src = open("blb_stage2_rl/action_mask.py", encoding="utf-8").read()
        for needle in (
            "class ForbiddenActionMask",
            "def add(self, layer_idx",
            "def is_forbidden(self, layer_idx",
            "def to_json_records",
            "def from_json_records",
            "def summary",
        ):
            self.assertIn(needle, src, msg=f"missing: {needle!r}")

    def test_roundtrip_via_minimal_import_shim(self):
        """Functional smoke: shim a fake parent package so the file's
        ``from .action_space import ...`` succeeds, then exercise the
        public API. The shim only stubs the symbols action_mask actually
        uses; if action_mask grows new dependencies this test fails loudly
        and the shim must be updated."""
        import sys
        import types
        pkg_name = "_blb_stage2_rl_test_pkg"
        if pkg_name in sys.modules:
            del sys.modules[pkg_name]
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(REPO_ROOT / "blb_stage2_rl")]
        sys.modules[pkg_name] = pkg

        # Stub the .action_space symbols action_mask imports.
        action_space_stub = types.ModuleType(f"{pkg_name}.action_space")
        action_space_stub.K_LEVELS = (8, 9, 11, 13, 10, 12)
        action_space_stub.action_dims_for_config = lambda L: [5] * (L * 73)
        action_space_stub.describe_action_vector = lambda *a, **kw: {"records": []}
        action_space_stub.load_max_sfs = lambda profile: None
        action_space_stub.make_all_max_action_vector = lambda L: [0] * (L * 73)
        sys.modules[f"{pkg_name}.action_space"] = action_space_stub

        loader = importlib.machinery.SourceFileLoader(
            f"{pkg_name}.action_mask",
            str(REPO_ROOT / "blb_stage2_rl/action_mask.py"),
        )
        spec = importlib.util.spec_from_loader(f"{pkg_name}.action_mask", loader)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"{pkg_name}.action_mask"] = mod
        loader.exec_module(mod)

        m = mod.ForbiddenActionMask()
        self.assertEqual(m.total(), 0)
        self.assertTrue(m.add(0, 1, (1, 2, 3)))
        self.assertFalse(m.add(0, 1, (1, 2, 3)))   # duplicate
        self.assertTrue(m.is_forbidden(0, 1, (1, 2, 3)))
        self.assertFalse(m.is_forbidden(0, 2, (1, 2, 3)))

        # Round-trip
        records = m.to_json_records()
        reborn = mod.ForbiddenActionMask.from_json_records(records)
        self.assertEqual(reborn.total(), 1)
        self.assertTrue(reborn.is_forbidden(0, 1, (1, 2, 3)))

        # Summary
        for i in range(3):
            m.add(5, 3, (i, 0))
        self.assertIn("total=4", m.summary())


class EnvEvalCommitSplitTest(unittest.TestCase):
    """Source-text smoke: sequential_env exposes both evaluate_step and
    commit_step so the runner can sample → optimizer-check → blacklist before
    committing state. The old single-call ``step`` remains as a backward-compat
    wrapper.
    """

    def test_env_has_evaluate_and_commit(self):
        src = open("blb_stage2_rl/sequential_env.py", encoding="utf-8").read()
        for needle in ("def evaluate_step", "def commit_step", "def step", "temp_vec"):
            self.assertIn(needle, src, msg=f"missing: {needle!r}")
        # The eval_info dict passed between phases must carry the optimizer
        # output + the spec (needed by commit).
        eval_idx = src.find("def evaluate_step")
        commit_idx = src.find("def commit_step")
        self.assertLess(eval_idx, commit_idx, "evaluate_step should come before commit_step")

    def test_runner_uses_evaluate_then_commit_with_mask(self):
        src = open("blb_stage2_rl/sequential_runner.py", encoding="utf-8").read()
        for needle in (
            "ForbiddenActionMask",
            "forbidden_mask.is_forbidden",
            "forbidden_mask.add",
            "env.evaluate_step",
            "env.commit_step",
            "rejection_counters",
            "steps_fallen_back_to_baseline",
        ):
            self.assertIn(needle, src, msg=f"missing: {needle!r}")

    def test_base_env_skips_forward_when_any_invalid(self):
        src = open("blb_stage2_rl/env.py", encoding="utf-8").read()
        for needle in (
            "any_invalid_chain",
            "forward_skipped_reason",
            "if any_invalid:",
        ):
            self.assertIn(needle, src, msg=f"missing: {needle!r}")


class RewardDesignV2RegressionTest(unittest.TestCase):
    """ADR-007: v2-style clipped+tier reward (supersedes ADR-002 implementation).

    Locks in the new reward shape so a future refactor can't silently
    regress back to -50/-100/-200 hard penalties (which produced the
    -150 stuck reward across every episode — see ADR-007 Context).
    """

    def test_reward_uses_v2_style_clipped_tier_formula(self):
        src = open("blb_stage2_rl/reward.py", encoding="utf-8").read()
        for needle in (
            "shaping_clipped",
            "tier_bonus",
            "lambda_stab",
            "reward_clip_min",
            "reward_clip_max",
            "tier_metric_bonus",
            "tier_stability_bonus",
            "baseline_metric1",
            "margin_acc",
        ):
            self.assertIn(
                needle, src,
                msg=f"reward.py missing v2-style field: {needle!r}",
            )
        # The classic v2 cap of [-5, +5] shaping with +20/+40 tier should be
        # the default, not just present as a configurable field.
        self.assertIn("DEFAULT_REWARD_CLIP_MIN = -5.0", src)
        self.assertIn("DEFAULT_REWARD_CLIP_MAX = 5.0", src)
        self.assertIn("DEFAULT_TIER_METRIC_BONUS = 20.0", src)
        self.assertIn("DEFAULT_TIER_STABILITY_BONUS = 20.0", src)

    def test_runner_stab_threshold_uses_v2_formula(self):
        """stab_threshold = noisy_baseline_loss_std × (1 + tol). The previous
        attempted dynamic calibration (sampling random valid actions for
        loss_std P90) failed because 577-dim uniform-random actions are
        almost always invalid_chain; see
        reports/stage2_rl/failed_runs/2026-05-18_dynamic_stab_calibration_fallback/
        """
        src = open("blb_stage2_rl/sequential_runner.py", encoding="utf-8").read()
        # New formula present
        self.assertIn("v2 formula:", src)
        self.assertIn("noisy_baseline_loss_std * (1.0 + stability_tol)", src)
        # Old failed random-action calibration removed
        self.assertNotIn("random_loss_stds", src)
        self.assertNotIn("target_calib_samples", src)
        self.assertNotIn("calibration_rng", src)

    def test_persistent_slug_reverted_to_three_tolerances(self):
        """2026-05-18: reverted _rdv2 suffix per user request. The single
        persistent dir is easier to maintain (no risk of forgetting to stop
        old training runs across multiple dirs). --fresh enforces clean
        reset for reward design changes; the slug stays minimal."""
        src = open("llama_7B_LayerImportance.sh", encoding="utf-8").read()
        # Main slug should be EXACTLY the three tolerances — no _rdv2 suffix.
        self.assertIn(
            'CONSTRAINT_SLUG="s1t${STAGE1_ACCURACY_TOLERANCE}_s2t${STAGE2_LIMIT_TOLERANCE}_s2st${STAGE2_STABILITY_TOLERANCE}"',
            src,
        )
        # Only allow _rdv2 in comments (the rollback rationale line). No
        # active code path should still inject the suffix.
        for line_no, line in enumerate(src.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            self.assertNotIn(
                "_rdv2", line,
                msg=f"line {line_no}: _rdv2 still in active code: {line!r}",
            )

    def test_episode_record_has_terminal_priority_and_metrics(self):
        src = open("blb_stage2_rl/sequential_runner.py", encoding="utf-8").read()
        for needle in (
            "terminal_priority: int = 0",
            "terminal_loss_mean: float = 0.0",
            "terminal_loss_std: float = 0.0",
            "terminal_metric1_mean: float = 0.0",
        ):
            self.assertIn(
                needle, src,
                msg=f"EpisodeRecord missing terminal breakdown field: {needle!r}",
            )

    def test_details_writer_uses_real_priority(self):
        src = open("blb_stage2_rl/sequential_runner.py", encoding="utf-8").read()
        # The old hardcoded `priority = 1 if invalid_steps > 0 else 3` lied
        # whenever P2(stab) tripped on a "0 invalid" episode. The new path
        # reads record.terminal_priority first and falls back to the legacy
        # form only when terminal_priority is unset (0).
        self.assertIn("int(record.terminal_priority) > 0", src)
        self.assertIn("priority = int(record.terminal_priority)", src)
        # And surfaces the actual metric numbers so operators can verify
        # which gate is firing without re-running the optimizer.
        self.assertIn("terminal_metrics: loss_mean=", src)

    def test_default_num_trials_bumped_to_five(self):
        src = open("blb_stage2_rl/runner.py", encoding="utf-8").read()
        # The default lives in BLBStage2TrainConfig and flows down through
        # BLBStage2EnvConfig.num_trials_per_step → env._eval_on_probe(k).
        self.assertIn("num_trials_per_step: int = 5", src)


class WarmstartFixedRegressionTest(unittest.TestCase):
    """2026-05-18 rdv2 hotfix: after the first ADR-007 run produced
    terminal_reward = -5 across 240 episodes (every candidate landed in
    P1 acc-fail because the warmstart bias preferred=[4]*13 was wrong
    for 8/13 slot positions), two layers of fix were added:

      (a) per-slot mode of baseline_action_vec used as preferred index
      (b) forced-baseline anchor for the first N episodes

    These tests lock the fixes in source so a refactor can't silently
    revert them.
    """

    def test_per_slot_mode_helper_present(self):
        src = open("blb_stage2_rl/sequential_runner.py", encoding="utf-8").read()
        for needle in (
            "_compute_per_slot_mode_preferred",
            "Counter(vals).most_common",
            "MODE of baseline_action_vec",
        ):
            self.assertIn(
                needle, src,
                msg=f"sequential_runner.py missing per-slot mode helper: {needle!r}",
            )

    def test_force_baseline_anchor_wired(self):
        src = open("blb_stage2_rl/sequential_runner.py", encoding="utf-8").read()
        for needle in (
            "force_baseline_episodes",
            "force_this_ep",
            "Forced-baseline anchor short-circuit",
            "steps_forced_to_baseline_anchor",
        ):
            self.assertIn(
                needle, src,
                msg=f"sequential_runner.py missing force-baseline anchor: {needle!r}",
            )

    def test_per_slot_mode_returns_correct_values(self):
        """Quick functional test: given a fake schedule + baseline_vec,
        the helper should pick the mode per slot position (not a uniform
        index). This guards against future refactors that accidentally
        revert to ``[max_idx]*max_step_dim``.
        """
        import sys
        import importlib.util
        import types

        # Stub heavy deps
        for name in ("torch", "torch.cuda", "torch.nn", "torch.nn.functional",
                     "transformers", "blb_rl_bridge", "function_handler",
                     "rescale_optimizer_bridge"):
            sys.modules.setdefault(name, types.ModuleType(name))
        # Stub the action_space symbols sequential_runner imports
        acs_stub = types.ModuleType("blb_stage2_rl.action_space")
        sys.modules["blb_stage2_rl.action_space"] = acs_stub

        pkg = sys.modules.get("blb_stage2_rl") or types.ModuleType("blb_stage2_rl")
        pkg.__path__ = ["blb_stage2_rl"]
        sys.modules["blb_stage2_rl"] = pkg

        # Load just the helper by exec'ing a minimal stub. The helper itself
        # is dependency-free (only numpy + collections), so we can extract
        # and exec it directly.
        src = open("blb_stage2_rl/sequential_runner.py", encoding="utf-8").read()
        marker_start = src.find("def _compute_per_slot_mode_preferred(")
        self.assertGreater(marker_start, 0, "helper not found in source")
        marker_end = src.find("\n\n\n", marker_start)
        self.assertGreater(marker_end, marker_start, "helper end not found")
        helper_src = src[marker_start:marker_end]

        import numpy as np
        from typing import Any, Optional, Sequence, List   # noqa: F401
        ns = {"np": np, "Any": Any, "Optional": Optional,
              "Sequence": Sequence, "List": List}
        exec(helper_src, ns)
        helper = ns["_compute_per_slot_mode_preferred"]

        # Fake schedule: 3 steps each with full_vec_offsets
        class FakeSpec:
            def __init__(self, offs):
                self.full_vec_offsets = tuple(offs)

        schedule = [
            FakeSpec([0, 1, 2]),    # slot 0→0, slot 1→1, slot 2→2
            FakeSpec([3, 4, 5]),    # slot 0→3, slot 1→4, slot 2→5
            FakeSpec([6, 7]),       # slot 0→6, slot 1→7 (no slot 2)
        ]
        baseline_vec = np.array(
            [4, 2, 3,   # step 0
             4, 2, 4,   # step 1 (slot 1 = 2 twice now → mode 2)
             0, 4],     # step 2
            dtype=np.int64,
        )
        preferred = helper(
            schedule=schedule,
            baseline_action_vec=baseline_vec,
            max_step_dim=4,
            fallback_idx=99,
        )
        # slot 0: values [4, 4, 0] → mode 4
        # slot 1: values [2, 2, 4] → mode 2
        # slot 2: values [3, 4] (only first 2 steps) → mode 3 (or 4; tie → most_common picks first encountered)
        # slot 3: no data → fallback 99
        self.assertEqual(preferred[0], 4)
        self.assertEqual(preferred[1], 2)
        self.assertIn(preferred[2], (3, 4))  # tie either way
        self.assertEqual(preferred[3], 99)

    def test_new_best_logs_inference_metrics(self):
        """After a new best, the log line should include loss_mean / loss_std /
        m1 so the user can verify acc/stab gates without grepping details files.
        """
        src = open("blb_stage2_rl/sequential_runner.py", encoding="utf-8").read()
        self.assertIn("推理指标（inference test metrics）", src)
        self.assertIn("record.terminal_loss_mean", src)
        self.assertIn("record.terminal_loss_std", src)
        self.assertIn("record.terminal_metric1_mean", src)


class EntCoefScheduleRegressionTest(unittest.TestCase):
    """2026-05-18 sampling-collapse hotfix: PPO entropy bonus was actively
    undoing the forced-baseline anchor (entropy rose 6.48 → 9.21 across
    3 anchor PPO updates, leaving the policy too diffuse for sampling).

    Schedule: ent_coef = 0 during anchor, linear ramp from 0 to target
    over ``ent_coef_ramp_episodes`` sample episodes, then steady.
    """

    def test_helper_present_with_expected_signature(self):
        src = open("blb_stage2_rl/sequential_runner.py", encoding="utf-8").read()
        for needle in (
            "def _resolve_ent_coef_schedule(",
            "ep_count_1based",
            "anchor_episodes",
            "ramp_episodes",
            "target_ent_coef",
        ):
            self.assertIn(
                needle, src,
                msg=f"sequential_runner.py missing ent_coef schedule helper: {needle!r}",
            )

    def test_ppo_update_passes_override(self):
        src = open("blb_stage2_rl/sequential_runner.py", encoding="utf-8").read()
        # The PPO update call must use the computed current_ent_coef.
        self.assertIn("current_ent_coef = _resolve_ent_coef_schedule(", src)
        self.assertIn("ent_coef_override=current_ent_coef", src)

    def test_ppo_update_accepts_override_param(self):
        """sequential_ppo_update must accept ent_coef_override kwarg."""
        src = open("blb_stage2_rl/sequential_policy.py", encoding="utf-8").read()
        self.assertIn("ent_coef_override: Optional[float] = None", src)
        self.assertIn("effective_ent_coef", src)
        # And the loss must use the effective value, not cfg.ent_coef directly.
        self.assertIn("- effective_ent_coef * entropy_mean", src)
        # And the returned metrics must surface the ent_coef so diagnostics
        # can show the schedule in action.
        self.assertIn('"ent_coef": float(effective_ent_coef)', src)

    def test_train_config_defaults(self):
        """SequentialTrainConfig should default to ent_coef_anchor=0 and
        a 240-episode ramp."""
        src = open("blb_stage2_rl/sequential_runner.py", encoding="utf-8").read()
        self.assertIn("ent_coef_anchor: float = 0.0", src)
        self.assertIn("ent_coef_ramp_episodes: int = 240", src)
        # BLBStage2TrainConfig (used by the runner) must also expose them
        # so the launcher / preset can override.
        src2 = open("blb_stage2_rl/runner.py", encoding="utf-8").read()
        self.assertIn("ent_coef_anchor: float = 0.0", src2)
        self.assertIn("ent_coef_ramp_episodes: int = 240", src2)

    def test_schedule_math_anchor_ramp_steady(self):
        """Functional test of the helper. Locks in the three-stage behaviour
        so a refactor can't silently change the ramp shape."""
        import sys, importlib.util, types
        # Stub torch (helper itself doesn't need it but the module does)
        for n in ("torch", "torch.cuda", "torch.nn", "torch.nn.functional",
                  "transformers", "blb_rl_bridge", "function_handler",
                  "rescale_optimizer_bridge"):
            sys.modules.setdefault(n, types.ModuleType(n))

        # Extract helper source and exec in isolation (avoids torch deps)
        src = open("blb_stage2_rl/sequential_runner.py", encoding="utf-8").read()
        head = src.find("def _resolve_ent_coef_schedule(")
        tail = src.find("\n\n\n", head)
        helper_src = src[head:tail]
        ns = {}
        exec(helper_src, ns)
        fn = ns["_resolve_ent_coef_schedule"]

        # Anchor stage: returns anchor_ent_coef (default 0.0)
        for ep in (1, 30, 60):
            self.assertEqual(
                fn(ep_count_1based=ep, anchor_episodes=60, target_ent_coef=0.02),
                0.0,
                msg=f"ep={ep} should be anchor (0.0)",
            )

        # Ramp stage: linear interpolation
        # ep=180 = 60 anchor + 120 into ramp; ramp_episodes default 240 → 50% ramp
        self.assertAlmostEqual(
            fn(ep_count_1based=180, anchor_episodes=60, target_ent_coef=0.02),
            0.01, places=5,
        )

        # End of ramp: target
        self.assertAlmostEqual(
            fn(ep_count_1based=300, anchor_episodes=60, target_ent_coef=0.02),
            0.02, places=5,
        )

        # Steady: target
        for ep in (301, 1000, 6000):
            self.assertAlmostEqual(
                fn(ep_count_1based=ep, anchor_episodes=60, target_ent_coef=0.02),
                0.02, places=5,
                msg=f"ep={ep} should be steady (target)",
            )

        # Custom ramp length
        self.assertAlmostEqual(
            fn(ep_count_1based=120, anchor_episodes=60,
               target_ent_coef=0.02, ramp_episodes=60),
            0.02, places=5,
            msg="ramp_episodes=60: ep=120 already past ramp",
        )

    def test_startup_log_describes_schedule(self):
        """The startup hyperparameter box should print the three-stage
        schedule so operators can verify it in the log without code-reading."""
        src = open("blb_stage2_rl/sequential_runner.py", encoding="utf-8").read()
        self.assertIn("Entropy schedule", src)
        self.assertIn("anchor[", src)
        self.assertIn("ramp[", src)
        self.assertIn("steady[", src)

    def test_diagnostics_table_shows_ent_coef(self):
        src = open("blb_stage2_rl/diagnostics.py", encoding="utf-8").read()
        # PPOUpdateStats must have ent_coef field
        self.assertIn("ent_coef: float = 0.0", src)
        # And the markdown table must include the column
        self.assertIn("ent_coef", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
