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


def _method_region_from_source(source: str, method_name: str) -> str:
    needle = f"    def {method_name}"
    next_needle = "\n    def "
    if needle not in source:
        needle = f"def {method_name}"
        next_needle = "\ndef "
    start = source.index(needle)
    next_method = source.find(next_needle, start + 1)
    if next_method == -1:
        next_method = len(source)
    return source[start:next_method]


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
                    terminal_priority=3,
                    terminal_bits_gain=float(ep * 100),
                    terminal_k_gain=float(ep % 3),
                    terminal_fusion_gain=float(ep),
                    terminal_cost_score=0.10,
                    terminal_p3_metric_margin_reward=0.05,
                    terminal_cost_fusion_bonus=0.20,
                    terminal_cost_truncation_bonus=0.30,
                    terminal_cost_bits_tiebreaker=0.04,
                    terminal_cost_truncation_step_gain=1.0,
                    fusion_action_steps=[
                        {
                            "step_idx": 0,
                            "layer_idx": 0,
                            "block_idx": 2,
                            "graph_key": "block2_mrpc",
                            "option_id": int(ep % 2),
                            "fusion_count": int(ep % 2),
                            "max_fusion": 1,
                            "k_index": 0,
                            "k_value": 13,
                            "valid": True,
                        }
                    ],
                    terminal_pareto_event_kind="frontier_expansion",
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
            "pareto_frontier.jsonl",
            "pareto_frontier.json",
            "pareto_frontier.html",
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
            self.assertIn("fusion_action_steps", r)
            self.assertEqual(r["fusion_action_steps"][0]["block_idx"], 2)
            self.assertIn("option_id", r["fusion_action_steps"][0])
            self.assertIn("k_value", r["fusion_action_steps"][0])

    def test_pareto_frontier_artifacts_and_dominance(self):
        run_dir = os.path.join(self.tmp, "run_pareto")
        self._synthesize_run(run_dir, n_episodes=24, seed=5)
        diag_dir = os.path.join(run_dir, "diagnostics")

        json_path = os.path.join(diag_dir, "pareto_frontier.json")
        jsonl_path = os.path.join(diag_dir, "pareto_frontier.jsonl")
        html_path = os.path.join(diag_dir, "pareto_frontier.html")
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]

        self.assertEqual(payload["schema_version"], "blb_stage2_pareto_frontier_v1")
        self.assertEqual(payload["count"], len(rows))
        self.assertTrue(rows)
        self.assertTrue(os.path.isfile(html_path))
        from blb_stage2_rl.candidate_store import action_hash

        for row in rows:
            self.assertEqual(
                row.get("terminal_pareto_action_hash"),
                action_hash(row.get("action_vec")),
            )
        self.assertIn("terminal_fusion_gain", payload["objectives"]["maximize"])
        self.assertIn("terminal_k_gain", payload["objectives"]["maximize"])
        self.assertIn("terminal_bits_gain", payload["objectives"]["maximize"])

        for i, a in enumerate(rows):
            self.assertIn("pareto_rank", a)
            self.assertEqual(a["terminal_priority"], 3)
            for j, b in enumerate(rows):
                if i == j:
                    continue
                self.assertFalse(
                    self.diag_mod._pareto_dominates(a, b),
                    msg=f"frontier row {i} dominates row {j}",
                )

    def test_top_candidates_use_stage1_reward_before_unbounded_p3_cost_rank(self):
        import numpy as np

        run_dir = os.path.join(self.tmp, "run_ranked_top")
        diag = self.diag_mod
        rec = diag.RLDiagnosticsRecorder(
            output_dir=run_dir,
            num_layers=12,
            num_action_slots=577,
            max_action_levels=6,
            top_k=2,
            log_fn=lambda *_: None,
        )

        def record(ep, total_reward, cost_rank, fusion_gain):
            rec.record_episode(
                episode_stats=diag.EpisodeStats(
                    episode=ep,
                    total_reward=float(total_reward),
                    terminal_reward=45.0,
                    per_step_sum=float(total_reward - 45.0),
                    valid_steps=59,
                    invalid_steps=0,
                    steps_taken=59,
                    total_bits=15000,
                    fusion_count=int(fusion_gain),
                    first_invalid_step=None,
                    first_invalid_block=None,
                    first_invalid_layer=None,
                    early_terminated=False,
                    terminal_priority=3,
                    terminal_cost_score=4.5,
                    terminal_cost_rank_score=float(cost_rank),
                    terminal_cost_rank_fusion=float(fusion_gain),
                    terminal_fusion_gain=float(fusion_gain),
                    terminal_k_gain=1.0,
                    terminal_bits_gain=300.0,
                ),
                full_action_vec=np.full(577, ep % 6, dtype=int),
                is_new_best=True,
                best_reward_so_far=float(total_reward),
            )

        record(1, 42.20, 6.0, 8.0)
        record(2, 42.00, 10.0, 14.0)
        rec.flush_periodic()

        top_path = os.path.join(run_dir, "diagnostics", "top_candidates.jsonl")
        rows = [
            json.loads(line)
            for line in Path(top_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(rows[0]["episode"], 1)
        self.assertGreater(rows[0]["total_reward"], rows[1]["total_reward"])

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
            "class StaticInvalidLevelMask",
            "def add_invalid",
            "class EmpiricalInvalidLevelMask",
            "def record_invalid",
            "def apply",
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

        static_mask = mod.StaticInvalidLevelMask()
        self.assertTrue(static_mask.add_invalid(0, 1, 0, 1, reason="bad_chain"))
        self.assertTrue(static_mask.add_invalid(0, 1, 1, 2, reason="bad_chain"))
        self.assertFalse(static_mask.add_invalid(0, 1, 1, 2, reason="duplicate"))
        base = [[True, True, True, True, True] for _ in range(3)]
        pruned_static = static_mask.apply(
            0,
            1,
            base,
            protected_actions=[(1, 0, 0)],
        )
        self.assertTrue(bool(pruned_static[0, 1]))  # protected baseline wins
        self.assertFalse(bool(pruned_static[1, 2]))
        self.assertEqual(static_mask.total_disabled(), 2)
        reborn_static = mod.StaticInvalidLevelMask.from_json_records(
            static_mask.to_json_records()
        )
        self.assertEqual(reborn_static.total_disabled(), static_mask.total_disabled())

        level_mask = mod.EmpiricalInvalidLevelMask(
            min_invalid_samples=2,
            min_invalid_rate=0.8,
            max_valid_samples=0,
        )
        level_mask.record_invalid(0, 1, (1, 2, 3))
        level_mask.record_invalid(0, 1, (1, 2, 4))
        base = [[True, True, True, True, True] for _ in range(3)]
        pruned = level_mask.apply(
            0,
            1,
            base,
            protected_actions=[(0, 0, 0)],
        )
        self.assertFalse(bool(pruned[0, 1]))
        self.assertFalse(bool(pruned[1, 2]))
        self.assertTrue(bool(pruned[2, 3]))
        self.assertTrue(bool(pruned[2, 4]))
        self.assertTrue(bool(pruned[0, 0]))
        records = level_mask.to_json_records()
        reborn_level = mod.EmpiricalInvalidLevelMask.from_json_records(records)
        reborn_level.min_invalid_samples = 2
        reborn_level.min_invalid_rate = 0.8
        reborn_level.max_valid_samples = 0
        self.assertEqual(reborn_level.total_disabled(), level_mask.total_disabled())


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
            "StaticInvalidLevelMask",
            "_precompute_static_invalid_level_mask",
            "_open_step_level_mask",
            "forbidden_mask.is_forbidden",
            "forbidden_mask.add",
            "env.evaluate_step",
            "env.commit_step",
            "rejection_counters",
            "steps_fallen_back_to_baseline",
            "empirical_invalid_mask",
            "EmpiricalInvalidLevelMask",
            "samples_rejected_by_optimizer",
            "rejection_optimizer_wall_seconds",
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


class MultiGpuProbeThroughputRegressionTest(unittest.TestCase):
    """Source-text locks for the four-GPU reward-probe throughput path."""

    def test_probe_runner_parallelizes_setup_and_clear(self):
        src = open("blb_stage2_rl/probe_runner.py", encoding="utf-8").read()
        for needle in (
            "def _for_each_worker",
            "threading.Thread(target=task",
            "self._for_each_worker(lambda w: w.install(decoded))",
            "self._for_each_worker(clear_one)",
            "enable_cuda_reward_probe_fast_math",
            "torch.backends.cuda.matmul.allow_tf32 = True",
        ):
            self.assertIn(needle, src, msg=f"probe_runner.py missing: {needle!r}")

    def test_probe_runner_caches_round_robin_trial_assignments(self):
        src = open("blb_stage2_rl/probe_runner.py", encoding="utf-8").read()
        for needle in (
            "from functools import lru_cache",
            "@lru_cache(maxsize=64)",
            "def _split_round_robin_cached",
            "assignments = _split_round_robin_cached(k, len(self.workers))",
        ):
            self.assertIn(needle, src, msg=f"probe_runner.py missing: {needle!r}")

    def test_probe_runner_aggregates_trial_results_in_preallocated_lists(self):
        src = open("blb_stage2_rl/probe_runner.py", encoding="utf-8").read()
        self.assertIn(
            "results_per_trial: List[Optional[Tuple[float, float, float]]] = [None] * k",
            src,
        )
        self.assertNotIn("results_per_trial: dict = {}", src)

    def test_env_has_persistent_install_and_timing_diagnostics(self):
        src = open("blb_stage2_rl/env.py", encoding="utf-8").read()
        for needle in (
            "persistent_probe_install: bool = False",
            "def clear_installed_blb",
            "cost_eval_wall_seconds",
            "probe_install_wall_seconds",
            "probe_clear_wall_seconds",
            "probe_install_skipped",
            "persistent_probe_install",
        ):
            self.assertIn(needle, src, msg=f"env.py missing: {needle!r}")

    def test_sequential_runner_enables_persistent_install_after_preflight(self):
        src = open("blb_stage2_rl/sequential_runner.py", encoding="utf-8").read()
        for needle in (
            "base_env.clear_installed_blb()",
            "base_env.env_cfg.persistent_probe_install = True",
            "Multi-GPU BLB install cache",
            "per_step_optimizer_wall_seconds",
            "terminal_probe_install_wall_seconds",
            "terminal_probe_clear_wall_seconds",
        ):
            self.assertIn(needle, src, msg=f"sequential_runner.py missing: {needle!r}")

    def test_rollout_policy_uses_causal_prefix_fast_path(self):
        policy_src = (REPO_ROOT / "blb_stage2_rl/sequential_policy.py").read_text(
            encoding="utf-8"
        )
        runner_src = (REPO_ROOT / "blb_stage2_rl/sequential_runner.py").read_text(
            encoding="utf-8"
        )
        parallel_runner_src = (
            REPO_ROOT / "blb_stage2_rl/parallel_runner.py"
        ).read_text(encoding="utf-8")
        for needle in (
            "truncate_to_current: bool = False",
            "truncate_seq_len: Optional[int] = None",
            "seq_len = int(current_step.detach().clamp(0, H - 1).item()) + 1",
            "if truncate_seq_len is not None:",
            "prev_actions = torch.zeros(B, seq_len, S",
            "register_buffer(\"_step_indices\"",
            "\"_level_indices\"",
            "_preferred_prior_template",
            "current_step.clamp(0, x.size(1) - 1)",
            "prev_action_embedding = nn.Embedding",
            "slot_head_weight = nn.Parameter",
            'torch.einsum("ba,sla->bsl"',
            "minibatch_size: int = 2048",
        ):
            self.assertIn(needle, policy_src, msg=f"sequential_policy.py missing: {needle!r}")
        self.assertIn("truncate_to_current=True", runner_src)
        self.assertIn("truncate_seq_len=int(spec.step_idx) + 1", runner_src)
        self.assertIn("truncate_seq_len=int(spec.step_idx) + 1", parallel_runner_src)
        self.assertIn("torch.inference_mode()", runner_src)
        self.assertIn("policy.eval()", runner_src)
        self.assertIn("policy_rollout_wall_seconds", runner_src)

    def test_sequential_rollout_buffer_packs_numpy_arrays_in_single_pass(self):
        policy_src = (REPO_ROOT / "blb_stage2_rl/sequential_policy.py").read_text(encoding="utf-8")
        for needle in (
            "def _pack_numpy_arrays",
            "for i, t in enumerate(buf):",
            "returns, advantages = _compute_gae_from_tensors(",
        ):
            self.assertIn(needle, policy_src, msg=f"sequential_policy.py missing: {needle!r}")
        for old_pattern in (
            "states = np.stack([t.state for t in self._buf])",
            "actions = np.stack([t.action for t in self._buf])",
            "old_values = np.array([t.value for t in self._buf], dtype=np.float32)",
            "returns, advantages = self.compute_gae(gamma=gamma, lam=lam)",
        ):
            self.assertNotIn(old_pattern, policy_src, msg=f"old multi-pass pack remains: {old_pattern!r}")

    def test_stage2_step_static_tensors_are_cached_per_schedule_device(self):
        runner_src = (REPO_ROOT / "blb_stage2_rl/sequential_runner.py").read_text(
            encoding="utf-8"
        )
        parallel_src = (REPO_ROOT / "blb_stage2_rl/parallel_runner.py").read_text(
            encoding="utf-8"
        )

        for needle in (
            "class _StepStaticTensors:",
            "def _get_cached_step_static_tensors(",
            "_stage2_static_step_tensor_cache",
            "torch.as_tensor(slot_mask_np, device=device).unsqueeze(0)",
            "torch.as_tensor(levels_np, device=device).unsqueeze(0)",
        ):
            self.assertIn(needle, runner_src, msg=f"sequential_runner.py missing: {needle!r}")
        for src_name, src in (
            ("sequential_runner.py", runner_src),
            ("parallel_runner.py", parallel_src),
        ):
            for needle in (
                "step_static_tensors = _get_cached_step_static_tensors(",
                "step_static = step_static_tensors[int(spec.step_idx)]",
                "slot_mask_t = step_static.slot_mask_t",
                "levels_t = step_static.levels_t",
            ):
                self.assertIn(needle, src, msg=f"{src_name} missing cache use: {needle!r}")
            for old_pattern in (
                "slot_mask_t = torch.from_numpy(slot_mask_np).to(device).unsqueeze(0)",
                "levels_t = torch.from_numpy(levels_np).to(device).unsqueeze(0)",
            ):
                self.assertNotIn(
                    old_pattern,
                    src,
                    msg=f"{src_name} still rebuilds static step tensors per step: {old_pattern!r}",
                )

    def test_stage2_fusion_action_level_masks_are_cached_per_device(self):
        runner_src = (REPO_ROOT / "blb_stage2_rl/sequential_runner.py").read_text(
            encoding="utf-8"
        )
        parallel_src = (REPO_ROOT / "blb_stage2_rl/parallel_runner.py").read_text(
            encoding="utf-8"
        )

        for needle in (
            "class _CachedFusionActionLevelMask:",
            "def _get_cached_fusion_action_level_mask(",
            "_stage2_fusion_action_level_mask_cache",
            "torch.as_tensor(mask_np, device=device).unsqueeze(0)",
        ):
            self.assertIn(needle, runner_src, msg=f"sequential_runner.py missing: {needle!r}")
        for src_name, src in (
            ("sequential_runner.py", runner_src),
            ("parallel_runner.py", parallel_src),
        ):
            self.assertIn(
                "_get_cached_fusion_action_level_mask(",
                src,
                msg=f"{src_name} does not use the cached fusion mask helper",
            )
        self.assertNotIn(
            "torch.from_numpy(action_level_mask_np).to(device).unsqueeze(0)",
            parallel_src,
            msg="parallel fusion rollout still copies action-level masks to GPU each step",
        )

    def test_stage2_parallel_rollout_batches_worker_scalar_sync_at_episode_end(self):
        parallel_src = (REPO_ROOT / "blb_stage2_rl/parallel_runner.py").read_text(
            encoding="utf-8"
        )
        collect_region = _method_region_from_source(parallel_src, "collect_fusion_episode")

        self.assertIn("def _materialize_transition_scalar_tensors(", parallel_src)
        self.assertIn("_materialize_transition_scalar_tensors(transitions)", collect_region)
        for needle in (
            "chosen_log_prob = lp_t.detach().reshape(())",
            "chosen_value = val_t.detach().reshape(())",
            "chosen_log_prob = log_prob_t.detach().reshape(())",
            "chosen_value = value_t.detach().reshape(())",
            "log_prob=chosen_log_prob",
            "value=chosen_value",
        ):
            self.assertIn(needle, collect_region, msg=f"parallel_runner.py missing: {needle!r}")
        for old_sync in (
            "chosen_log_prob = float(lp_t.item())",
            "chosen_value = float(val_t.item())",
            "chosen_log_prob = float(log_prob_t.item())",
            "chosen_value = float(value_t.item())",
            "log_prob=float(chosen_log_prob)",
            "value=float(chosen_value)",
        ):
            self.assertNotIn(
                old_sync,
                collect_region,
                msg=f"parallel rollout still syncs scalar per step: {old_sync!r}",
            )

    def test_sequential_stage2_rollout_defers_logprob_value_sync_until_buffer_pack(self):
        policy_src = (REPO_ROOT / "blb_stage2_rl/sequential_policy.py").read_text(encoding="utf-8")
        runner_src = (REPO_ROOT / "blb_stage2_rl/sequential_runner.py").read_text(encoding="utf-8")
        add_region = _method_region_from_source(policy_src, "add")
        to_tensors_region = _method_region_from_source(policy_src, "to_tensors")

        self.assertIn("def _pack_transition_scalar_tensors", policy_src)
        self.assertIn("def _compute_gae_from_tensors", policy_src)
        self.assertIn("log_probs_t = _pack_transition_scalar_tensors(", to_tensors_region)
        self.assertIn("old_values_t = _pack_transition_scalar_tensors(", to_tensors_region)
        self.assertIn("returns, advantages = _compute_gae_from_tensors(", to_tensors_region)
        self.assertIn("log_prob=log_prob.detach().reshape(())", add_region)
        self.assertIn("value=value.detach().reshape(())", add_region)
        self.assertNotIn("log_prob=float(log_prob)", add_region)
        self.assertNotIn("value=float(value)", add_region)
        for old_sync in (
            "chosen_log_prob = float(lp_t.item())",
            "chosen_value = float(val_t.item())",
            "chosen_log_prob = float(log_prob_t.item())",
            "chosen_value = float(value_t.item())",
        ):
            self.assertNotIn(old_sync, runner_src)
        self.assertIn("chosen_log_prob = lp_t.detach().reshape(())", runner_src)
        self.assertIn("chosen_log_prob = log_prob_t.detach().reshape(())", runner_src)

    def test_sequential_ppo_update_reuses_device_minibatch_indices_per_epoch(self):
        policy_src = (REPO_ROOT / "blb_stage2_rl/sequential_policy.py").read_text(encoding="utf-8")
        update_region = _method_region_from_source(policy_src, "sequential_ppo_update")

        self.assertIn("epoch_indices = torch.randperm(n, device=device)", update_region)
        self.assertIn("mb = epoch_indices[start:end]", update_region)
        self.assertNotIn("np.random.shuffle(indices)", update_region)
        self.assertNotIn("epoch_indices = torch.from_numpy(indices).long().to(device)", update_region)
        self.assertNotIn("mb = torch.from_numpy(indices[start:end]).long().to(device)", update_region)

    def test_sequential_ppo_update_reuses_minibatch_gathers(self):
        policy_src = (REPO_ROOT / "blb_stage2_rl/sequential_policy.py").read_text(encoding="utf-8")
        update_region = _method_region_from_source(policy_src, "sequential_ppo_update")

        self.assertIn("mb_slot_masks = slot_masks.index_select(0, mb)", update_region)
        self.assertIn("mb_levels = levels.index_select(0, mb)", update_region)
        self.assertIn("mb_level_masks = (", update_region)
        self.assertIn("mb_prior_scales = prior_scales.index_select(0, mb)", update_region)
        self.assertIn("mb_slot_masks_float = mb_slot_masks.float()", update_region)
        self.assertIn("mb_levels_float = mb_levels.float().clamp_min(1.0)", update_region)
        self.assertEqual(update_region.count("slot_masks.index_select(0, mb)"), 1)
        self.assertEqual(update_region.count("levels.index_select(0, mb)"), 1)
        self.assertEqual(update_region.count("level_masks.index_select(0, mb)"), 1)
        self.assertEqual(update_region.count("prior_scales.index_select(0, mb)"), 1)

    def test_sequential_ppo_update_accumulates_metrics_on_device(self):
        policy_src = (REPO_ROOT / "blb_stage2_rl/sequential_policy.py").read_text(encoding="utf-8")
        update_region = _method_region_from_source(policy_src, "sequential_ppo_update")

        self.assertIn("metrics_sum_t = {", update_region)
        self.assertIn('"policy_loss": torch.zeros((), device=device)', update_region)
        self.assertIn("clip_frac_t = ((torch.abs(ratio - 1.0) > cfg.clip_range).float()).mean()", update_region)
        self.assertIn("approx_kl_t = (old_lp - new_log_probs).mean()", update_region)
        self.assertIn('metrics_sum_t["policy_loss"] += policy_loss.detach()', update_region)
        self.assertIn('metrics_sum_t["clip_fraction"] += clip_frac_t.detach()', update_region)
        self.assertIn('epoch_avg_kl_t = metrics_sum_t["approx_kl"] / float(n_seen)', update_region)
        self.assertIn('float((metrics_sum_t["policy_loss"] / n_mb).item())', update_region)
        self.assertNotIn('metrics_sum["policy_loss"] += float(policy_loss.item())', update_region)
        self.assertNotIn('metrics_sum["value_loss"] += float(value_loss.item())', update_region)
        self.assertNotIn('metrics_sum["entropy"] += float(entropy_mean.item())', update_region)
        self.assertNotIn("mean().item()", update_region)

    def test_sequential_ppo_update_batches_nonfinite_checks_before_cpu_sync(self):
        policy_src = (REPO_ROOT / "blb_stage2_rl/sequential_policy.py").read_text(encoding="utf-8")
        update_region = _method_region_from_source(policy_src, "sequential_ppo_update")

        self.assertIn("finite_checks = torch.stack(", update_region)
        self.assertIn("torch.isfinite(t).all().reshape(())", update_region)
        self.assertIn("if not bool(finite_checks.all().item()):", update_region)
        self.assertNotIn(
            "all(bool(torch.isfinite(t).all().item()) for t in finite_tensors)",
            update_region,
        )

    def test_sequential_advantage_norm_avoids_scalar_sync(self):
        policy_src = (REPO_ROOT / "blb_stage2_rl/sequential_policy.py").read_text(encoding="utf-8")
        robust_region = _method_region_from_source(policy_src, "_robust_normalize_advantages")
        update_region = _method_region_from_source(policy_src, "sequential_ppo_update")

        self.assertIn("mad_ok = torch.isfinite(mad) & (mad > 1e-8)", robust_region)
        self.assertIn("adv = torch.where(mad_ok, clipped_adv, adv)", robust_region)
        self.assertIn("std_ok = torch.isfinite(std) & (std > 1e-8)", robust_region)
        self.assertIn("adv = torch.where(std_ok, normalized_adv, adv)", robust_region)
        self.assertNotIn(".item()", robust_region)
        self.assertIn("normalized_advantages = (advantages - torch.mean(advantages)) / (std + 1e-8)", update_region)
        self.assertIn("advantages = torch.where(std_ok, normalized_advantages, advantages)", update_region)
        self.assertNotIn("torch.isfinite(std).item()", update_region)

    def test_legacy_rollout_buffer_packs_numpy_arrays_in_single_pass(self):
        policy_src = (REPO_ROOT / "blb_stage2_rl/policy.py").read_text(encoding="utf-8")
        to_tensors_region = _method_region_from_source(policy_src, "to_tensors")
        for needle in (
            "def _pack_rollout_samples",
            "for i, sample in enumerate(samples):",
            "states, actions, rewards = _pack_rollout_samples(",
            "def _pack_rollout_scalar_tensors",
        ):
            self.assertIn(needle, policy_src, msg=f"policy.py missing: {needle!r}")
        for old_pattern in (
            "states = np.stack([s.state for s in self._samples])",
            "actions = np.stack([s.action for s in self._samples])",
            "log_probs = np.array([s.log_prob for s in self._samples], dtype=np.float32)",
            "rewards = np.array([s.reward for s in self._samples], dtype=np.float32)",
            "values = np.array([s.value for s in self._samples], dtype=np.float32)",
        ):
            self.assertNotIn(old_pattern, to_tensors_region, msg=f"old multi-pass pack remains: {old_pattern!r}")

    def test_legacy_stage2_rollout_defers_logprob_value_sync_until_buffer_pack(self):
        policy_src = (REPO_ROOT / "blb_stage2_rl/policy.py").read_text(encoding="utf-8")
        runner_src = (REPO_ROOT / "blb_stage2_rl/runner.py").read_text(encoding="utf-8")
        add_region = _method_region_from_source(policy_src, "add")
        to_tensors_region = _method_region_from_source(policy_src, "to_tensors")

        self.assertIn("def _pack_rollout_scalar_tensors", policy_src)
        self.assertIn("log_probs_t = _pack_rollout_scalar_tensors(", to_tensors_region)
        self.assertIn("old_values_t = _pack_rollout_scalar_tensors(", to_tensors_region)
        self.assertIn("advantages = returns - old_values_t", to_tensors_region)
        self.assertIn("if not any(torch.is_tensor(value) for value in values):", policy_src)
        self.assertIn("torch.as_tensor([float(value) for value in values]", policy_src)
        self.assertIn("log_prob=log_prob.detach().reshape(())", add_region)
        self.assertIn("value=value.detach().reshape(())", add_region)
        self.assertNotIn("log_prob=float(log_prob)", add_region)
        self.assertNotIn("value=float(value)", add_region)
        self.assertNotIn("log_prob = float(log_prob_t.item())", runner_src)
        self.assertNotIn("value = float(value_t.item())", runner_src)
        self.assertIn("log_prob = log_prob_t.detach().reshape(())", runner_src)
        self.assertIn("value = value_t.detach().reshape(())", runner_src)

    def test_legacy_ppo_update_reuses_device_minibatch_indices_per_epoch(self):
        policy_src = (REPO_ROOT / "blb_stage2_rl/policy.py").read_text(encoding="utf-8")
        update_region = _method_region_from_source(policy_src, "ppo_update")

        self.assertIn("epoch_indices = torch.randperm(n, device=device)", update_region)
        self.assertIn("mb_idx_t = epoch_indices[start:end]", update_region)
        self.assertNotIn("np.random.shuffle(indices)", update_region)
        self.assertNotIn("epoch_indices = torch.from_numpy(indices).long().to(device)", update_region)
        self.assertNotIn("mb_idx = indices[start:end]", update_region)
        self.assertNotIn("torch.from_numpy(mb_idx).long().to(device)", update_region)

    def test_legacy_ppo_update_accumulates_metrics_on_device(self):
        policy_src = (REPO_ROOT / "blb_stage2_rl/policy.py").read_text(encoding="utf-8")
        update_region = _method_region_from_source(policy_src, "ppo_update")

        self.assertIn("metrics_sum_t = {", update_region)
        self.assertIn('"policy_loss": torch.zeros((), device=device)', update_region)
        self.assertIn("clip_frac_t = ((torch.abs(ratio - 1.0) > cfg.clip_range).float()).mean()", update_region)
        self.assertIn('metrics_sum_t["policy_loss"] += policy_loss.detach()', update_region)
        self.assertIn('metrics_sum_t["clip_fraction"] += clip_frac_t.detach()', update_region)
        self.assertIn('float((metrics_sum_t["policy_loss"] / n_mb).item())', update_region)
        self.assertNotIn('metrics_sum["policy_loss"] += float(policy_loss.item())', update_region)
        self.assertNotIn('metrics_sum["value_loss"] += float(value_loss.item())', update_region)
        self.assertNotIn('metrics_sum["entropy"] += float(entropy_mean.item())', update_region)
        self.assertNotIn("mean().item()", update_region)

    def test_stage1_ppo_return_normalization_stays_on_device(self):
        evaluator_src = (REPO_ROOT / "layer_importance_evaluator.py").read_text(encoding="utf-8")
        update_region = _method_region_from_source(evaluator_src, "ppo_update_gtrxl")

        self.assertIn("returns_normalized = self.return_normalizer.normalize(returns).to(", update_region)
        self.assertIn("values_normalized = self.return_normalizer.normalize(values).to(", update_region)
        self.assertNotIn("returns.cpu().numpy()", update_region)
        self.assertNotIn("values.cpu().numpy()", update_region)
        self.assertNotIn("returns_normalized = torch.tensor(", update_region)
        self.assertNotIn("values_normalized = torch.tensor(", update_region)

    def test_stage1_ppo_computes_gae_batch_on_device(self):
        evaluator_src = (REPO_ROOT / "layer_importance_evaluator.py").read_text(encoding="utf-8")
        update_region = _method_region_from_source(evaluator_src, "ppo_update_gtrxl")
        batch_gae_region = _method_region_from_source(evaluator_src, "compute_gae_batch")

        self.assertIn("advantages, returns = self.compute_gae_batch(rewards, values, dones)", update_region)
        self.assertNotIn("all_advantages", update_region)
        self.assertNotIn("all_returns", update_region)
        self.assertNotIn("rewards[i].cpu().numpy()", update_region)
        self.assertNotIn("values[i].cpu().numpy()", update_region)
        self.assertNotIn("dones[i].cpu().numpy()", update_region)
        self.assertIn("torch.zeros_like(rewards", batch_gae_region)
        self.assertIn("values[:, t + 1]", batch_gae_region)
        self.assertIn("advantages[:, t] = gae", batch_gae_region)

    def test_stage1_ppo_epoch_indices_are_created_on_device(self):
        evaluator_src = (REPO_ROOT / "layer_importance_evaluator.py").read_text(encoding="utf-8")
        update_region = _method_region_from_source(evaluator_src, "ppo_update_gtrxl")

        self.assertIn("ep_indices = torch.randperm(n_eps, device=device)", update_region)
        self.assertNotIn("ep_indices = torch.randperm(n_eps)\n", update_region)

    def test_running_mean_std_updates_torch_tensor_without_full_cpu_copy(self):
        evaluator_src = (REPO_ROOT / "layer_importance_evaluator.py").read_text(encoding="utf-8")
        update_region = _method_region_from_source(evaluator_src, "update")

        self.assertIn("x_detached = x.detach()", update_region)
        self.assertIn("batch_mean = float(x_detached.mean().item())", update_region)
        self.assertIn("batch_var = float(x_detached.var(unbiased=False).item())", update_region)
        self.assertIn("batch_count = int(x_detached.numel())", update_region)
        self.assertNotIn("x = x.detach().cpu().numpy()", update_region)

    def test_sequential_running_mean_std_updates_torch_tensor_without_full_cpu_copy(self):
        policy_src = (REPO_ROOT / "blb_stage2_rl/sequential_policy.py").read_text(encoding="utf-8")
        update_region = _method_region_from_source(policy_src, "update")

        self.assertIn("values_detached = values.detach()", update_region)
        self.assertIn("stats = torch.stack(", update_region)
        self.assertIn("batch_count = int(values_detached.numel())", update_region)
        self.assertIn("batch_mean, batch_var = (float(x) for x in stats", update_region)
        self.assertNotIn("values = values.detach().cpu().numpy()", update_region)

    def test_stage1_entropy_recovery_avoids_loss_path_item_sync(self):
        evaluator_src = (REPO_ROOT / "layer_importance_evaluator.py").read_text(encoding="utf-8")
        update_region = _method_region_from_source(evaluator_src, "ppo_update_gtrxl")

        self.assertIn("mean_entropy_detached = mean_entropy.detach()", update_region)
        self.assertIn("entropy_deficit = torch.relu(", update_region)
        self.assertIn(
            "effective_entropy_coef = entropy_coef + _rl_opt_entropy_recovery_mul() * entropy_deficit",
            update_region,
        )
        self.assertNotIn("if mean_entropy.item() < _entropy_lb:", update_region)
        self.assertNotIn("mean_entropy.item()", update_region)

    def test_stage1_ppo_return_metrics_sync_once_after_minibatches(self):
        evaluator_src = (REPO_ROOT / "layer_importance_evaluator.py").read_text(encoding="utf-8")
        update_region = _method_region_from_source(evaluator_src, "ppo_update_gtrxl")

        self.assertIn("last_policy_loss_t = None", update_region)
        self.assertIn("last_value_loss_t = None", update_region)
        self.assertIn("last_entropy_t = None", update_region)
        self.assertIn("last_policy_loss_t = policy_loss.detach()", update_region)
        self.assertIn("last_value_loss_t = value_loss.detach()", update_region)
        self.assertIn("last_entropy_t = mean_entropy.detach()", update_region)
        self.assertIn("float(last_policy_loss_t.item())", update_region)
        self.assertIn("float(last_value_loss_t.item())", update_region)
        self.assertIn("float(last_entropy_t.item())", update_region)
        self.assertNotIn("last_policy_loss = policy_loss.item()", update_region)
        self.assertNotIn("last_value_loss = value_loss.item()", update_region)
        self.assertNotIn("last_entropy = mean_entropy.item()", update_region)

    def test_stage1_ppo_kl_early_stop_accumulates_on_device_until_epoch_end(self):
        evaluator_src = (REPO_ROOT / "layer_importance_evaluator.py").read_text(encoding="utf-8")
        update_region = _method_region_from_source(evaluator_src, "ppo_update_gtrxl")

        self.assertIn("epoch_kl_acc_t = torch.zeros((), device=device)", update_region)
        self.assertIn("approx_kl_t = (mb_old_lp_flat - new_logprobs_flat).mean()", update_region)
        self.assertIn("epoch_kl_acc_t += approx_kl_t.detach()", update_region)
        self.assertIn("avg_kl_t = epoch_kl_acc_t / float(epoch_kl_count)", update_region)
        self.assertIn("float(avg_kl_t.item())", update_region)
        self.assertNotIn("(mb_old_lp_flat - new_logprobs_flat).mean().item()", update_region)
        self.assertNotIn("epoch_kl_acc += approx_kl", update_region)

    def test_legacy_action_mask_validation_avoids_gpu_scalar_sync_for_numpy_masks(self):
        policy_src = (REPO_ROOT / "blb_stage2_rl/policy.py").read_text(encoding="utf-8")
        mask_region = _method_region_from_source(policy_src, "_mask_logits_for_slot")

        self.assertIn("raw_arr = np.asarray(raw, dtype=bool).reshape(-1)", mask_region)
        self.assertIn("if not bool(raw_arr.any()):", mask_region)
        self.assertIn("torch.as_tensor(raw_arr, dtype=torch.bool, device=logits.device)", mask_region)
        self.assertNotIn("mask.any().item()", mask_region)

    def test_action_dist_avoids_gpu_sync_for_zero_exploration_floor(self):
        policy_src = (REPO_ROOT / "blb_stage2_rl/sequential_policy.py").read_text(encoding="utf-8")
        action_dist_region = _method_region_from_source(policy_src, "_action_dist")
        setter_region = _method_region_from_source(policy_src, "set_slot_exploration_epsilon")

        self.assertIn("self._slot_exploration_enabled = False", policy_src)
        self.assertIn("self._slot_exploration_enabled = bool(", setter_region)
        self.assertIn("if not self._slot_exploration_enabled:", action_dist_region)
        self.assertNotIn(".item()", action_dist_region)

    def test_truncated_policy_forward_matches_full_causal_prefix(self):
        for name in ("torch", "torch.cuda", "torch.nn", "torch.nn.functional"):
            sys.modules.pop(name, None)
        try:
            import numpy as np
            import torch
        except Exception as exc:
            self.skipTest(f"torch/numpy unavailable: {exc}")

        policy_mod = _load_module_standalone(
            "blb_stage2_rl/sequential_policy.py",
            "sequential_policy_truncated_forward_test",
        )
        BLBStage2SequentialPolicy = policy_mod.BLBStage2SequentialPolicy
        SequentialPolicyConfig = policy_mod.SequentialPolicyConfig

        torch.manual_seed(7)
        horizon = 10
        max_step_dim = 6
        state_dim = 4 + horizon + 5 + 1 + horizon * max_step_dim + horizon * 3
        cfg = SequentialPolicyConfig(
            state_dim=state_dim,
            max_step_dim=max_step_dim,
            max_num_levels=6,
            horizon=horizon,
            num_layers=3,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=128,
            dropout=0.0,
        )
        policy = BLBStage2SequentialPolicy(cfg).eval()
        state = torch.zeros(1, state_dim)
        current_step = 6
        state[:, :4] = torch.tensor([[0.5, 0.25, 0.5, 0.1]])
        state[:, 4 + current_step] = 1.0
        cursor = 4 + horizon + 5 + 1
        rng = np.random.default_rng(123)
        # Fill the entire history area, including future slots. Causal masking
        # must make the current-step output independent of future entries.
        actions = rng.integers(0, 6, size=(horizon, max_step_dim)).astype(np.float32)
        state[:, cursor: cursor + horizon * max_step_dim] = torch.from_numpy(
            (actions.reshape(-1) / 8.0).astype(np.float32)
        )
        cursor += horizon * max_step_dim
        signals = rng.normal(size=(horizon, 3)).astype(np.float32)
        state[:, cursor: cursor + horizon * 3] = torch.from_numpy(signals.reshape(-1))

        with torch.inference_mode():
            full_logits, full_value = policy.forward(state, truncate_to_current=False)
            trunc_logits, trunc_value = policy.forward(state, truncate_to_current=True)

        self.assertTrue(torch.allclose(trunc_logits, full_logits, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(trunc_value, full_value, atol=1e-5, rtol=1e-5))

        preferred = [idx % cfg.max_num_levels for idx in range(cfg.max_step_dim)]
        policy.apply_preferred_per_step_bias(preferred, gain=1.0)
        template = policy._preferred_prior_template.detach().cpu()
        self.assertEqual(float(template.sum().item()), float(cfg.max_step_dim))
        for slot_idx, level_idx in enumerate(preferred):
            self.assertEqual(float(template[slot_idx, level_idx].item()), 1.0)


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

    def test_runner_stability_logs_all_three_std_thresholds(self):
        """Stability gates must be visible for loss, m1, and m2.

        The reward path already combines metric1_std, metric2_std, and loss_std.
        Keep the runner metadata/log contract aligned so reports do not imply
        that stability is loss-only.
        """
        src = open("blb_stage2_rl/sequential_runner.py", encoding="utf-8").read()
        self.assertIn("stab_threshold_m1", src)
        self.assertIn("stab_threshold_m2", src)
        self.assertIn("std_thresholds(loss/m1/m2)", src)
        self.assertIn("metric1_std_threshold", src)
        self.assertIn("metric2_std_threshold", src)
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

    def test_stage2_stability_tolerance_allows_large_slack(self):
        """ADR-011 uses --stage2-stability-tolerance 5.0 for a 5x std gate.

        Unlike metric degradation tolerances, stability tolerance is a
        multiplier in threshold = baseline_std * tol, so values above 1
        are valid and must pass launcher validation.
        """
        src = open("llama_7B_LayerImportance.sh", encoding="utf-8").read()
        self.assertIn("可设 5.0 表示 5×/500%", src)
        self.assertIn("Stage-2 稳定性约束倍率", src)
        self.assertIn(
            'is_pos_num "$STAGE2_STABILITY_TOLERANCE"',
            src,
        )
        self.assertNotIn(
            "--stage2-stability-tolerance 必须 < 1",
            src,
        )

    def test_episode_record_has_terminal_priority_and_metrics(self):
        src = open("blb_stage2_rl/sequential_runner.py", encoding="utf-8").read()
        for needle in (
            "terminal_priority: int = 0",
            "terminal_loss_mean: float = 0.0",
            "terminal_loss_std: float = 0.0",
            "terminal_metric1_mean: float = 0.0",
            "terminal_metric2_mean: float = 0.0",
            "terminal_metric1_std: float = 0.0",
            "terminal_metric2_std: float = 0.0",
            "terminal_stab_violation: float = 0.0",
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

    def _exec_runner_helpers(self, *names):
        """Extract dependency-light helpers from sequential_runner without
        importing the torch-heavy module on local developer machines."""
        import ast
        import math
        import numpy as np
        from pathlib import Path
        from typing import Any, Optional, Sequence, Set, List   # noqa: F401

        src = Path("blb_stage2_rl/sequential_runner.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        wanted = set(names)
        body = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in wanted
        ]
        self.assertEqual(
            {node.name for node in body},
            wanted,
            msg=f"missing helper(s): {sorted(wanted - {node.name for node in body})}",
        )
        mod = ast.Module(body=body, type_ignores=[])
        ast.fix_missing_locations(mod)
        ns = {
            "Any": Any,
            "Optional": Optional,
            "Sequence": Sequence,
            "Set": Set,
            "List": List,
            "np": np,
            "math": math,
            "K_LEVELS": (8, 9, 11, 13, 10, 12),
        }
        exec(compile(mod, "<sequential_runner_helpers>", "exec"), ns)
        return ns

    def _exec_runner_items(self, *names):
        """Extract dependency-light functions/classes from sequential_runner."""
        import ast
        import math
        import numpy as np
        from dataclasses import dataclass, field
        from pathlib import Path
        from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple   # noqa: F401

        src = Path("blb_stage2_rl/sequential_runner.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        wanted = set(names)
        body = [
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted
        ]
        self.assertEqual(
            {node.name for node in body},
            wanted,
            msg=f"missing helper/class: {sorted(wanted - {node.name for node in body})}",
        )
        mod = ast.Module(body=body, type_ignores=[])
        ast.fix_missing_locations(mod)
        ns = {
            "Any": Any,
            "Dict": Dict,
            "List": List,
            "Mapping": Mapping,
            "Optional": Optional,
            "Sequence": Sequence,
            "Set": Set,
            "Tuple": Tuple,
            "np": np,
            "math": math,
            "dataclass": dataclass,
            "field": field,
        }
        exec(compile(mod, "<sequential_runner_items>", "exec"), ns)
        return ns

    def test_sequential_force_anchor_honors_explicit_warmstart_only(self):
        ns = self._exec_runner_helpers("_resolve_sequential_force_baseline_episodes")
        helper = ns["_resolve_sequential_force_baseline_episodes"]

        class Cfg:
            rollout_size = 60
            total_episodes = 6000
            force_baseline_episodes = 0
            warmstart_anchor_episodes = 80

        self.assertEqual(helper(Cfg()), 80)

        Cfg.force_baseline_episodes = 160
        self.assertEqual(helper(Cfg()), 160)

        Cfg.force_baseline_episodes = 0
        Cfg.warmstart_anchor_episodes = None
        self.assertEqual(helper(Cfg()), 0)

    def test_continuous_reward_does_not_bypass_force_anchor(self):
        src = open("blb_stage2_rl/sequential_runner.py", encoding="utf-8").read()
        self.assertNotIn(
            "0 if _continuous else _resolve_sequential_force_baseline_episodes",
            src,
        )
        self.assertIn(
            "_force_baseline_episodes = _resolve_sequential_force_baseline_episodes(train_cfg)",
            src,
        )

    def test_stage1_aligned_reward_bypasses_fusion_scaffolds(self):
        src = open("blb_stage2_rl/sequential_runner.py", encoding="utf-8").read()
        self.assertIn(
            "_fc_curriculum_on = False if _continuous else",
            src,
        )
        self.assertIn(
            "fusion_probe_interval=(0 if _continuous else",
            src,
        )
        self.assertIn(
            "fusion_exploration_epsilon=(0.0 if _continuous else",
            src,
        )

    def test_noisy_metric_threshold_uses_relative_tolerance_without_probe_guard(self):
        ns = self._exec_runner_helpers("_noisy_metric_threshold_from_baseline")
        helper = ns["_noisy_metric_threshold_from_baseline"]

        threshold = helper(
            noisy_baseline_metric=0.8727,
            tolerance=0.001,
        )
        self.assertAlmostEqual(threshold, 0.8727 * 0.999, places=7)
        self.assertGreater(threshold, 0.8718)
        self.assertGreater(threshold, 0.31)

    def test_baseline_prior_schedule_decays_to_zero(self):
        ns = self._exec_runner_helpers("_resolve_baseline_prior_scale")
        helper = ns["_resolve_baseline_prior_scale"]

        self.assertAlmostEqual(helper(0, anchor_episodes=120), 8.0)
        self.assertAlmostEqual(helper(120, anchor_episodes=120), 8.0)
        self.assertAlmostEqual(helper(1000, anchor_episodes=120), 6.0)
        self.assertAlmostEqual(helper(5000, anchor_episodes=120), 3.0)
        self.assertAlmostEqual(helper(15000, anchor_episodes=120), 0.0)
        self.assertAlmostEqual(helper(60000, anchor_episodes=120), 0.0)

    def test_step_level_mask_keeps_unselected_slots_at_baseline(self):
        ns = self._exec_runner_helpers(
            "_near_baseline_level_indices",
            "_default_step_level_mask",
            "_build_step_level_mask",
        )
        build_mask = ns["_build_step_level_mask"]
        near = ns["_near_baseline_level_indices"]

        self.assertEqual(set(near(kind="K", baseline_idx=3, dim=6, radius=2)), {1, 2, 3, 4, 5})

        class FakeSpec:
            slot_dims = (5, 6, 3)
            slot_kinds = ("F", "K", "M")
            full_vec_offsets = (0, 1, 2)

        import numpy as np

        baseline = np.array([4, 3, 2], dtype=np.int64)
        mask = build_mask(
            spec=FakeSpec(),
            baseline_action_vec=baseline,
            selected_full_offsets={1, 2},
            max_step_dim=4,
            max_num_levels=6,
            radius=2,
        )
        self.assertEqual(mask.shape, (4, 6))
        self.assertEqual(set(np.flatnonzero(mask[0]).tolist()), {4})
        self.assertEqual(set(np.flatnonzero(mask[1]).tolist()), {1, 2, 3, 4, 5})
        self.assertEqual(set(np.flatnonzero(mask[2]).tolist()), {0, 1, 2})
        self.assertFalse(mask[3].any())

        baseline_only = build_mask(
            spec=FakeSpec(),
            baseline_action_vec=baseline,
            selected_full_offsets=set(),
            max_step_dim=4,
            max_num_levels=6,
            radius=2,
        )
        self.assertEqual(set(np.flatnonzero(baseline_only[0]).tolist()), {4})
        self.assertEqual(set(np.flatnonzero(baseline_only[1]).tolist()), {3})
        self.assertEqual(set(np.flatnonzero(baseline_only[2]).tolist()), {2})

    def test_guarded_radius2_waits_for_stall_and_healthy_history(self):
        ns = self._exec_runner_items(
            "GuardedRadius2Decision",
            "OffsetEmpiricalStats",
            "GuardedRadius2Controller",
        )
        Controller = ns["GuardedRadius2Controller"]

        controller = Controller(
            enabled=True,
            min_episode=1060,
            stall_window=600,
            health_window=100,
            max_mutations=4,
            episode_fraction=1.0,
            cooldown_episodes=300,
            min_radius1_successes=3,
        )
        import numpy as np

        # Not enough history and not past min_episode.
        decision = controller.decide(absolute_episode_idx=1000, rng=np.random.default_rng(0))
        self.assertFalse(decision.active)
        self.assertEqual(decision.mode, "radius1")

        for ep in range(600):
            event = "dominated"
            controller.record_episode(
                absolute_episode_idx=ep,
                selected_offsets={1, 2, 3},
                radius=1,
                terminal_priority=3,
                invalid_steps=0,
                early_terminated=False,
                terminal_stab_violation=0.0,
                terminal_loss_mean=0.3,
                terminal_pareto_event_kind=event,
            )
        decision = controller.decide(absolute_episode_idx=1060, rng=np.random.default_rng(1))
        self.assertTrue(decision.active)
        self.assertEqual(decision.mode, "guarded_radius2")
        self.assertEqual(decision.radius, 2)
        self.assertLessEqual(decision.mutation_count, 4)
        self.assertGreaterEqual(decision.safe_offset_count, 3)

        # A recent frontier expansion means the frontier is not stalled.
        controller.record_episode(
            absolute_episode_idx=1060,
            selected_offsets={1, 2},
            radius=1,
            terminal_priority=3,
            invalid_steps=0,
            early_terminated=False,
            terminal_stab_violation=0.0,
            terminal_loss_mean=0.3,
            terminal_pareto_event_kind="frontier_expansion",
        )
        decision = controller.decide(absolute_episode_idx=1061, rng=np.random.default_rng(2))
        self.assertFalse(decision.active)
        self.assertIn("frontier", decision.reason)

    def test_guarded_radius2_cooldown_after_radius2_failure(self):
        ns = self._exec_runner_items(
            "GuardedRadius2Decision",
            "OffsetEmpiricalStats",
            "GuardedRadius2Controller",
        )
        Controller = ns["GuardedRadius2Controller"]
        controller = Controller(
            enabled=True,
            min_episode=1060,
            stall_window=3,
            health_window=3,
            max_mutations=4,
            episode_fraction=1.0,
            cooldown_episodes=300,
            min_radius1_successes=3,
        )
        import numpy as np

        for ep in range(3):
            controller.record_episode(
                absolute_episode_idx=ep,
                selected_offsets={10, 11},
                radius=1,
                terminal_priority=3,
                invalid_steps=0,
                early_terminated=False,
                terminal_stab_violation=0.0,
                terminal_loss_mean=0.3,
                terminal_pareto_event_kind="dominated",
            )
        self.assertTrue(controller.decide(absolute_episode_idx=1060, rng=np.random.default_rng(3)).active)

        controller.record_episode(
            absolute_episode_idx=1060,
            selected_offsets={10},
            radius=2,
            terminal_priority=1,
            invalid_steps=0,
            early_terminated=False,
            terminal_stab_violation=0.0,
            terminal_loss_mean=0.3,
            terminal_pareto_event_kind="excluded",
        )
        decision = controller.decide(absolute_episode_idx=1061, rng=np.random.default_rng(4))
        self.assertFalse(decision.active)
        self.assertGreaterEqual(decision.cooldown_remaining, 299)
        self.assertIn("cooldown", decision.reason)

    def test_sequential_ppo_buffer_carries_action_level_mask(self):
        from pathlib import Path

        policy_src = Path("blb_stage2_rl/sequential_policy.py").read_text(encoding="utf-8")
        runner_src = Path("blb_stage2_rl/sequential_runner.py").read_text(encoding="utf-8")
        for needle in (
            "action_level_mask: np.ndarray",
            "action_level_mask=action_level_mask_np",
            "level_masks",
            "action_level_mask: Optional[torch.Tensor] = None",
        ):
            self.assertIn(needle, policy_src + runner_src, msg=f"missing mask wiring: {needle!r}")

    def test_10k_curve_knobs_are_launcher_visible(self):
        """The 10k research loop needs to tune exploration without editing
        Python for every server run."""
        from pathlib import Path

        launcher = Path("llama_7B_LayerImportance.sh").read_text(encoding="utf-8")
        rl_tune = Path("rl_tune.py").read_text(encoding="utf-8")
        evaluator = Path("layer_importance_evaluator.py").read_text(encoding="utf-8")
        runner = Path("blb_stage2_rl/runner.py").read_text(encoding="utf-8")
        combined = "\n".join([launcher, rl_tune, evaluator, runner])
        for needle in (
            "--blb-v3-warmstart-neighbor-ramp-episodes",
            "--blb-v3-warmstart-neighbor-max-mutations",
            "--blb-v3-warmstart-neighbor-max-radius",
            "--blb-v3-warmstart-neighbor-sampling",
            "--blb-v3-guarded-radius2-enabled",
            "--blb-v3-guarded-radius2-min-episode",
            "--blb-v3-guarded-radius2-stall-window",
            "--blb-v3-guarded-radius2-max-mutations",
            "--blb-v3-guarded-radius2-episode-fraction",
            "--blb-v3-guarded-radius2-cooldown-episodes",
            "--blb-v3-warmstart-bias-gain",
            "--blb-v3-ent-coef",
            "--blb-v3-ent-coef-ramp-episodes",
            "--blb-v3-static-invalid-level-mask-enabled",
            "blb_v3_warmstart_neighbor_ramp_episodes",
            "blb_v3_warmstart_neighbor_max_mutations",
            "blb_v3_warmstart_neighbor_max_radius",
            "blb_v3_guarded_radius2_enabled",
            "blb_v3_guarded_radius2_min_episode",
            "blb_v3_guarded_radius2_stall_window",
            "blb_v3_guarded_radius2_max_mutations",
            "blb_v3_guarded_radius2_episode_fraction",
            "blb_v3_guarded_radius2_cooldown_episodes",
            "blb_v3_warmstart_bias_gain",
            "blb_v3_ent_coef",
            "blb_v3_ent_coef_ramp_episodes",
            "blb_v3_static_invalid_level_mask_enabled",
            '("ent_coef_ramp_episodes", "blb_v3_ent_coef_ramp_episodes")',
        ):
            self.assertIn(needle, combined, msg=f"missing 10k curve knob: {needle!r}")

    def test_episode_jsonl_carries_terminal_health_fields(self):
        """10k online monitoring must not depend on grepping details text for
        terminal loss/P1/safe-neighbor health."""
        from pathlib import Path

        diagnostics = Path("blb_stage2_rl/diagnostics.py").read_text(encoding="utf-8")
        runner_src = Path("blb_stage2_rl/sequential_runner.py").read_text(encoding="utf-8")
        for needle in (
            "terminal_priority: int = 0",
            "terminal_loss_mean: float = 0.0",
            "terminal_loss_std: float = 0.0",
            "terminal_metric1_mean: float = 0.0",
            "terminal_metric2_mean: float = 0.0",
            "terminal_metric1_std: float = 0.0",
            "terminal_metric2_std: float = 0.0",
            "terminal_stab_violation: float = 0.0",
            "terminal_bits_gain: float = 0.0",
            "terminal_k_gain: float = 0.0",
            "terminal_fusion_gain: float = 0.0",
            "terminal_p3_metric_margin_reward: float = 0.0",
            "terminal_cost_fusion_bonus: float = 0.0",
            "terminal_cost_truncation_bonus: float = 0.0",
            "terminal_cost_bits_tiebreaker: float = 0.0",
            "terminal_cost_truncation_step_gain: float = 0.0",
            "terminal_cost_rank_score: float = 0.0",
            "terminal_cost_rank_fusion: float = 0.0",
            "terminal_cost_rank_truncation: float = 0.0",
            "terminal_cost_rank_bits: float = 0.0",
            "terminal_pareto_event_kind: str = \"\"",
            "safe_neighbor_active: bool = False",
            "exploration_mode: str = \"\"",
            "guarded_radius2_active: bool = False",
            "guarded_radius2_recent_frontier_expansions: int = 0",
            "guarded_radius2_recent_duplicate_rate: float = 0.0",
            "guarded_radius2_recent_dominated_rate: float = 0.0",
            "guarded_radius2_cooldown_remaining: int = 0",
            "samples_rejected_by_mask: int = 0",
            "samples_rejected_by_optimizer: int = 0",
            "steps_fallen_back_to_baseline: int = 0",
            "forbidden_mask_total: int = 0",
            "empirical_invalid_level_disabled: int = 0",
            "rejection_optimizer_wall_seconds: float = 0.0",
            "terminal_priority=int(record.terminal_priority)",
            "terminal_metric2_mean=float(record.terminal_metric2_mean)",
            "terminal_stab_violation=float(record.terminal_stab_violation)",
            "terminal_bits_gain=float(record.terminal_bits_gain)",
            "terminal_p3_metric_margin_reward=float(",
            "terminal_cost_fusion_bonus=float(record.terminal_cost_fusion_bonus)",
            "terminal_cost_truncation_bonus=float(record.terminal_cost_truncation_bonus)",
            "terminal_cost_bits_tiebreaker=float(record.terminal_cost_bits_tiebreaker)",
            "terminal_cost_truncation_step_gain=float(",
            "terminal_cost_rank_score=float(record.terminal_cost_rank_score)",
            "terminal_cost_rank_fusion=float(record.terminal_cost_rank_fusion)",
            "terminal_cost_rank_truncation=float(record.terminal_cost_rank_truncation)",
            "terminal_cost_rank_bits=float(record.terminal_cost_rank_bits)",
            "terminal_pareto_event_kind=str(record.terminal_pareto_event_kind)",
            "safe_neighbor_mutation_count=int(record.safe_neighbor_mutation_count)",
            "exploration_mode=str(record.exploration_mode)",
            "guarded_radius2_active=bool(record.guarded_radius2_active)",
            "samples_rejected_by_optimizer=int(record.samples_rejected_by_optimizer)",
            "rejection_optimizer_wall_seconds=float(record.rejection_optimizer_wall_seconds)",
        ):
            self.assertIn(needle, diagnostics + runner_src, msg=f"missing JSONL health field: {needle!r}")

    def test_p3_best_rank_uses_stage1_reward_before_unbounded_cost_rank(self):
        ns = self._exec_runner_items(
            "EpisodeRecord",
            "_episode_best_rank_key",
        )
        Record = ns["EpisodeRecord"]
        rank_key = ns["_episode_best_rank_key"]

        capped_lower_cost = Record(
            episode_idx=1,
            total_reward=42.20,
            terminal_reward=45.0,
            per_step_reward_sum=-2.80,
            invalid_steps=0,
            early_terminated=False,
            steps_taken=59,
            terminal_priority=3,
            terminal_cost_score=4.5,
            terminal_cost_rank_score=6.0,
            terminal_fusion_gain=8.0,
            terminal_k_gain=0.5,
            terminal_bits_gain=300.0,
        )
        capped_higher_cost = Record(
            episode_idx=2,
            total_reward=42.00,
            terminal_reward=45.0,
            per_step_reward_sum=-3.00,
            invalid_steps=0,
            early_terminated=False,
            steps_taken=59,
            terminal_priority=3,
            terminal_cost_score=4.5,
            terminal_cost_rank_score=10.0,
            terminal_fusion_gain=14.0,
            terminal_k_gain=1.2,
            terminal_bits_gain=500.0,
        )
        p2_high_cost = Record(
            episode_idx=3,
            total_reward=60.0,
            terminal_reward=25.0,
            per_step_reward_sum=35.0,
            invalid_steps=0,
            early_terminated=False,
            steps_taken=59,
            terminal_priority=2,
            terminal_cost_score=4.5,
            terminal_cost_rank_score=999.0,
        )

        self.assertGreater(
            rank_key(capped_lower_cost),
            rank_key(capped_higher_cost),
        )
        capped_same_reward_higher_cost = Record(
            episode_idx=4,
            total_reward=42.20,
            terminal_reward=45.0,
            per_step_reward_sum=-2.80,
            invalid_steps=0,
            early_terminated=False,
            steps_taken=59,
            terminal_priority=3,
            terminal_cost_score=4.5,
            terminal_cost_rank_score=10.0,
            terminal_fusion_gain=14.0,
            terminal_k_gain=1.2,
            terminal_bits_gain=500.0,
        )
        self.assertGreater(
            rank_key(capped_same_reward_higher_cost),
            rank_key(capped_lower_cost),
        )
        self.assertGreater(
            rank_key(capped_lower_cost),
            rank_key(p2_high_cost),
        )

    def test_final_strict_selection_uses_next_feasible_top_candidate(self):
        ns = self._exec_runner_items(
            "EpisodeRecord",
            "_episode_best_rank_key",
            "_stage2_record_loss_ok",
            "_stage2_record_strict_feasible",
            "_select_stage2_strict_feasible_best_record",
        )
        Record = ns["EpisodeRecord"]
        select = ns["_select_stage2_strict_feasible_best_record"]

        loss_threshold = 0.3661778037
        rank_best_loss_fail = Record(
            episode_idx=10,
            total_reward=0.497,
            terminal_reward=0.497,
            per_step_reward_sum=0.0,
            invalid_steps=0,
            early_terminated=False,
            steps_taken=59,
            terminal_priority=3,
            terminal_loss_mean=0.3662689477,
            terminal_cost_rank_score=3930.0,
            terminal_fusion_gain=23.0,
        )
        next_feasible = Record(
            episode_idx=11,
            total_reward=0.494,
            terminal_reward=0.494,
            per_step_reward_sum=0.0,
            invalid_steps=0,
            early_terminated=False,
            steps_taken=59,
            terminal_priority=3,
            terminal_loss_mean=0.3655472547,
            terminal_cost_rank_score=3900.0,
            terminal_fusion_gain=22.0,
        )
        p2_candidate = Record(
            episode_idx=12,
            total_reward=0.8,
            terminal_reward=0.8,
            per_step_reward_sum=0.0,
            invalid_steps=0,
            early_terminated=False,
            steps_taken=59,
            terminal_priority=2,
            terminal_loss_mean=0.360,
            terminal_cost_rank_score=9999.0,
        )

        selected = select(
            [p2_candidate, next_feasible, rank_best_loss_fail],
            loss_threshold=loss_threshold,
            top_n=20,
        )
        self.assertIs(selected, next_feasible)

    def test_final_strict_selection_respects_top_n(self):
        ns = self._exec_runner_items(
            "EpisodeRecord",
            "_episode_best_rank_key",
            "_stage2_record_loss_ok",
            "_stage2_record_strict_feasible",
            "_select_stage2_strict_feasible_best_record",
        )
        Record = ns["EpisodeRecord"]
        select = ns["_select_stage2_strict_feasible_best_record"]

        failing_top = Record(
            episode_idx=1,
            total_reward=1.0,
            terminal_reward=1.0,
            per_step_reward_sum=0.0,
            invalid_steps=0,
            early_terminated=False,
            steps_taken=59,
            terminal_priority=3,
            terminal_loss_mean=0.40,
            terminal_cost_rank_score=100.0,
        )
        feasible_outside_top1 = Record(
            episode_idx=2,
            total_reward=0.5,
            terminal_reward=0.5,
            per_step_reward_sum=0.0,
            invalid_steps=0,
            early_terminated=False,
            steps_taken=59,
            terminal_priority=3,
            terminal_loss_mean=0.35,
            terminal_cost_rank_score=90.0,
        )

        self.assertIsNone(
            select(
                [failing_top, feasible_outside_top1],
                loss_threshold=0.36,
                top_n=1,
            )
        )
        self.assertIs(
            select(
                [failing_top, feasible_outside_top1],
                loss_threshold=0.36,
                top_n=2,
            ),
            feasible_outside_top1,
        )

    def test_first10k_server_defaults_avoid_radius2_collapse_region(self):
        """The first failed 10k run showed P1s when radius reached 2."""
        from pathlib import Path

        src = Path("scripts/stage2_first10k_server_run.sh").read_text(encoding="utf-8")
        for needle in (
            'ANCHOR_EPISODES="${ANCHOR_EPISODES:-60}"',
            'NEIGHBOR_RAMP="${NEIGHBOR_RAMP:-1800}"',
            'NEIGHBOR_MAX_MUTATIONS="${NEIGHBOR_MAX_MUTATIONS:-12}"',
            'NEIGHBOR_MAX_RADIUS="${NEIGHBOR_MAX_RADIUS:-1}"',
            'ENT_COEF="${ENT_COEF:-0.06}"',
            'ENT_RAMP="${ENT_RAMP:-600}"',
            'WARMSTART_BIAS_GAIN="${WARMSTART_BIAS_GAIN:-1.2}"',
            'GUARDED_RADIUS2_ENABLED="${GUARDED_RADIUS2_ENABLED:-1}"',
            'GUARDED_RADIUS2_MIN_EPISODE="${GUARDED_RADIUS2_MIN_EPISODE:-1060}"',
            'GUARDED_RADIUS2_STALL_WINDOW="${GUARDED_RADIUS2_STALL_WINDOW:-600}"',
            'GUARDED_RADIUS2_MAX_MUTATIONS="${GUARDED_RADIUS2_MAX_MUTATIONS:-4}"',
            'GUARDED_RADIUS2_EPISODE_FRACTION="${GUARDED_RADIUS2_EPISODE_FRACTION:-0.15}"',
            'GUARDED_RADIUS2_COOLDOWN_EPISODES="${GUARDED_RADIUS2_COOLDOWN_EPISODES:-300}"',
            '--blb-v3-warmstart-bias-gain "$WARMSTART_BIAS_GAIN"',
            '--blb-v3-guarded-radius2-enabled "$GUARDED_RADIUS2_ENABLED"',
            '--blb-v3-guarded-radius2-min-episode "$GUARDED_RADIUS2_MIN_EPISODE"',
            '--blb-v3-guarded-radius2-stall-window "$GUARDED_RADIUS2_STALL_WINDOW"',
            '--blb-v3-guarded-radius2-max-mutations "$GUARDED_RADIUS2_MAX_MUTATIONS"',
            '--blb-v3-guarded-radius2-episode-fraction "$GUARDED_RADIUS2_EPISODE_FRACTION"',
            '--blb-v3-guarded-radius2-cooldown-episodes "$GUARDED_RADIUS2_COOLDOWN_EPISODES"',
            '--expected-reward-devices "$REWARD_DEVICES"',
        ):
            self.assertIn(needle, src)
        self.assertNotIn('NEIGHBOR_MAX_RADIUS="${NEIGHBOR_MAX_RADIUS:-2}"', src)
        self.assertNotIn('NEIGHBOR_MAX_RADIUS="${NEIGHBOR_MAX_RADIUS:-3}"', src)

    def test_first10k_server_run_aborts_on_git_pull_failure(self):
        """Server runs must not continue experiments from a stale HEAD."""
        from pathlib import Path

        src = Path("scripts/stage2_first10k_server_run.sh").read_text(encoding="utf-8")
        self.assertIn("timeout 180 git pull --ff-only", src)
        self.assertIn('echo "[abort] git pull failed or timed out (rc=$PULL_RC); refusing to run on stale HEAD."', src)
        self.assertIn('exit "$PULL_RC"', src)
        self.assertNotIn("continuing with current HEAD", src)

    def test_first10k_monitor_tolerates_sparse_loss_cap_spikes(self):
        """User criterion allows sparse spikes; bursts or frequent caps still fail."""
        import argparse

        monitor = _load_module_standalone(
            "scripts/stage2_first10k_monitor.py", "stage2_first10k_monitor_test",
        )

        def write_case(loss_cap_episodes):
            tmp = Path(tempfile.mkdtemp(prefix="first10k_monitor_"))
            rows = []
            for ep in range(220):
                rows.append({
                    "episode": ep,
                    "total_reward": 40.0,
                    "terminal_reward": 40.0,
                    "terminal_priority": 3,
                    "terminal_loss_mean": 100.0 if ep in loss_cap_episodes else 0.34,
                    "terminal_loss_std": 0.003,
                    "terminal_metric1_mean": 0.87,
                    "valid_steps": 59,
                    "invalid_steps": 0,
                    "total_bits": 14770,
                    "safe_neighbor_active": ep >= 120,
                    "safe_neighbor_mutation_count": 4,
                    "safe_neighbor_radius": 1,
                })
            (tmp / "episodes.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            (tmp / "nvidia.csv").write_text(
                "timestamp,gpu_idx,util_pct,mem_used_mib\n",
                encoding="utf-8",
            )
            return tmp

        def summary_for(tmp):
            args = argparse.Namespace(
                phase="live",
                artifact_dir=str(tmp),
                stage2_noise=str(tmp),
                nvidia_log=str(tmp / "nvidia.csv"),
                planned=10000,
                anchor=120,
                rollout=60,
                horizon=59,
                k_trials=5,
                probe_size=256,
                expected_reward_devices="",
            )
            return monitor.build_summary(args)

        isolated_dir = write_case({150})
        sparse_dir = write_case({150, 210})
        burst_dir = write_case({150, 151})
        frequent_dir = write_case({150, 170, 190, 205, 219})
        try:
            isolated = summary_for(isolated_dir)
            sparse = summary_for(sparse_dir)
            burst = summary_for(burst_dir)
            frequent = summary_for(frequent_dir)
            self.assertFalse(isolated["hard_failures"], isolated["hard_failures"])
            self.assertTrue(isolated["warnings"], isolated)
            self.assertFalse(sparse["hard_failures"], sparse["hard_failures"])
            self.assertTrue(sparse["warnings"], sparse)
            self.assertTrue(
                any("terminal_loss_mean collapse cap" in x for x in burst["hard_failures"]),
                burst["hard_failures"],
            )
            self.assertTrue(
                any("terminal_loss_mean collapse cap" in x for x in frequent["hard_failures"]),
                frequent["hard_failures"],
            )
        finally:
            shutil.rmtree(isolated_dir, ignore_errors=True)
            shutil.rmtree(sparse_dir, ignore_errors=True)
            shutil.rmtree(burst_dir, ignore_errors=True)
            shutil.rmtree(frequent_dir, ignore_errors=True)

    def test_first10k_monitor_uses_post_anchor_p12_rate_threshold(self):
        import argparse

        monitor = _load_module_standalone(
            "scripts/stage2_first10k_monitor.py", "stage2_first10k_monitor_p12_test",
        )

        def write_case(priority_tail):
            tmp = Path(tempfile.mkdtemp(prefix="first10k_monitor_p12_"))
            rows = []
            for ep in range(120):
                rows.append({
                    "episode": ep,
                    "total_reward": 40.0,
                    "terminal_reward": 40.0,
                    "terminal_priority": 3,
                    "terminal_loss_mean": 0.34,
                    "terminal_loss_std": 0.003,
                    "terminal_metric1_mean": 0.87,
                    "valid_steps": 59,
                    "invalid_steps": 0,
                    "total_bits": 14770,
                    "safe_neighbor_active": False,
                })
            for idx, prio in enumerate(priority_tail):
                ep = 120 + idx
                rows.append({
                    "episode": ep,
                    "total_reward": 37.0 if prio == 3 else -4.0,
                    "terminal_reward": 37.0 if prio == 3 else -5.0,
                    "terminal_priority": int(prio),
                    "terminal_loss_mean": 0.34,
                    "terminal_loss_std": 0.003,
                    "terminal_metric1_mean": 0.87,
                    "valid_steps": 59,
                    "invalid_steps": 0,
                    "total_bits": 14770,
                    "safe_neighbor_active": True,
                })
            (tmp / "episodes.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            (tmp / "nvidia.csv").write_text(
                "timestamp,gpu_idx,util_pct,mem_used_mib\n",
                encoding="utf-8",
            )
            return tmp

        def summary_for(tmp):
            args = argparse.Namespace(
                phase="live",
                artifact_dir=str(tmp),
                stage2_noise=str(tmp),
                nvidia_log=str(tmp / "nvidia.csv"),
                planned=10000,
                anchor=120,
                rollout=60,
                horizon=59,
                k_trials=4,
                probe_size=256,
                expected_reward_devices="",
                max_post_anchor_p12_rate=0.30,
                min_post_anchor_p12_rate_samples=100,
            )
            return monitor.build_summary(args)

        at_threshold = write_case(([1] * 20) + ([2] * 10) + ([3] * 70))
        above_threshold = write_case(([1] * 20) + ([2] * 11) + ([3] * 69))
        try:
            ok = summary_for(at_threshold)
            bad = summary_for(above_threshold)
            self.assertFalse(ok["hard_failures"], ok["hard_failures"])
            self.assertEqual(ok["priority"]["post_anchor_p12_count"], 30)
            self.assertAlmostEqual(ok["priority"]["post_anchor_p12_rate"], 0.30)
            self.assertTrue(
                any("P1/P2" in item for item in ok["warnings"]),
                ok["warnings"],
            )
            self.assertTrue(
                any("P1/P2 rate exceeded" in item for item in bad["hard_failures"]),
                bad["hard_failures"],
            )
            self.assertEqual(bad["priority"]["post_anchor_p12_count"], 31)
        finally:
            shutil.rmtree(at_threshold, ignore_errors=True)
            shutil.rmtree(above_threshold, ignore_errors=True)

    def test_first10k_server_run_passes_p12_rate_threshold(self):
        src = Path("scripts/stage2_first10k_server_run.sh").read_text(encoding="utf-8")
        self.assertIn('MAX_POST_ANCHOR_P12_RATE="${MAX_POST_ANCHOR_P12_RATE:-0.30}"', src)
        self.assertIn('P12_RATE_MIN_POST_ANCHOR="${P12_RATE_MIN_POST_ANCHOR:-100}"', src)
        self.assertIn("--max-post-anchor-p12-rate", src)
        self.assertIn("--min-post-anchor-p12-rate-samples", src)
        self.assertIn("--blb-v3-static-invalid-level-mask-enabled 1", src)

    def test_first10k_server_run_can_skip_pull_only_on_verified_head(self):
        src = Path("scripts/stage2_first10k_server_run.sh").read_text(encoding="utf-8")
        self.assertIn('ALLOW_VERIFIED_HEAD_WITHOUT_PULL="${ALLOW_VERIFIED_HEAD_WITHOUT_PULL:-0}"', src)
        self.assertIn('EXPECTED_SOURCE_COMMIT="${EXPECTED_SOURCE_COMMIT:-}"', src)
        self.assertIn('[ "$ALLOW_VERIFIED_HEAD_WITHOUT_PULL" = "1" ]', src)
        self.assertIn('current_head="$(git rev-parse HEAD)"', src)
        self.assertIn('expected_full="$(git rev-parse "$EXPECTED_SOURCE_COMMIT^{commit}")"', src)
        self.assertIn('refusing to skip git pull', src)
        self.assertIn('git pull failed or timed out', src)

    def test_fast_multi_actor_reward_mode_is_configurable_and_safe(self):
        runner = Path("blb_stage2_rl/runner.py").read_text(encoding="utf-8")
        seq_runner = Path("blb_stage2_rl/sequential_runner.py").read_text(encoding="utf-8")
        launcher = Path("llama_7B_LayerImportance.sh").read_text(encoding="utf-8")
        combined = "\n".join([runner, seq_runner, launcher])
        for needle in (
            "fast_reward_mode_enabled",
            "online_num_trials_per_step",
            "terminal_eval_batch_size",
            "promotion_validation_trials",
            "promotion_margin_window",
            "--blb-v3-fast-reward-mode-enabled",
            "--blb-v3-online-k-trials",
            "--blb-v3-terminal-eval-batch-size",
            "--blb-v3-promotion-validation-trials",
        ):
            self.assertIn(needle, combined, msg=f"missing fast reward knob: {needle!r}")

    def test_probe_runner_can_evaluate_distinct_actions_on_distinct_workers(self):
        src = Path("blb_stage2_rl/probe_runner.py").read_text(encoding="utf-8")
        for needle in (
            "run_action_trials_once",
            "decoded_by_trial",
            "worker.install(decoded)",
            "worker.run_trial",
            "multi_action",
        ):
            self.assertIn(needle, src, msg=f"missing multi-action probe support: {needle!r}")

    def test_sequential_fast_mode_defers_terminal_reward_and_revalidates_promotions(self):
        src = Path("blb_stage2_rl/sequential_runner.py").read_text(encoding="utf-8")
        for needle in (
            "defer_terminal_forward",
            "pending_terminal_drafts",
            "flush_pending_terminal_drafts",
            "terminal_buffer_index",
            "promotion_validation",
            "cached_reward_hit",
            "validation_required",
        ):
            self.assertIn(needle, src, msg=f"missing deferred terminal fast-mode path: {needle!r}")

    def test_first10k_monitor_checks_all_expected_reward_gpus(self):
        import argparse

        monitor = _load_module_standalone(
            "scripts/stage2_first10k_monitor.py", "stage2_first10k_monitor_gpu_test",
        )
        tmp = Path(tempfile.mkdtemp(prefix="first10k_monitor_gpu_"))
        try:
            row = {
                "episode": 0,
                "total_reward": 40.0,
                "terminal_reward": 40.0,
                "terminal_priority": 3,
                "terminal_loss_mean": 0.34,
                "terminal_loss_std": 0.003,
                "terminal_metric1_mean": 0.87,
                "terminal_metric2_mean": 0.87,
                "valid_steps": 59,
                "invalid_steps": 0,
                "total_bits": 14770,
                "safe_neighbor_active": True,
                "terminal_probe_devices": ["cuda:0", "cuda:1"],
                "terminal_probe_trial_counts": [2, 2],
            }
            (tmp / "episodes.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            (tmp / "nvidia.csv").write_text(
                "timestamp,gpu_idx,util_pct,mem_used_mib\n"
                "t,0,50,1000\n"
                "t,1,50,1000\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                phase="final",
                artifact_dir=str(tmp),
                stage2_noise=str(tmp),
                nvidia_log=str(tmp / "nvidia.csv"),
                planned=1,
                anchor=0,
                rollout=60,
                horizon=59,
                k_trials=4,
                probe_size=256,
                expected_reward_devices="0,1,2,3",
            )
            missing = monitor.build_summary(args)
            self.assertTrue(
                any("Reward probe devices" in x for x in missing["hard_failures"]),
                missing["hard_failures"],
            )

            row["terminal_probe_devices"] = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
            row["terminal_probe_trial_counts"] = [1, 1, 1, 1]
            (tmp / "episodes.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            (tmp / "nvidia.csv").write_text(
                "timestamp,gpu_idx,util_pct,mem_used_mib\n"
                "t,0,50,1000\n"
                "t,1,50,1000\n"
                "t,2,50,1000\n"
                "t,3,50,1000\n",
                encoding="utf-8",
            )
            ok = monitor.build_summary(args)
            self.assertFalse(ok["hard_failures"], ok["hard_failures"])
            self.assertEqual(
                ok["reward_probe"]["observed_trial_splits"],
                [[1, 1, 1, 1]],
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_sequential_ppo_update_replays_stored_action_level_mask(self):
        import sys

        torch_mod = sys.modules.get("torch")
        if torch_mod is not None and not hasattr(getattr(torch_mod, "nn", None), "Module"):
            for name in list(sys.modules):
                if name == "torch" or name.startswith("torch."):
                    del sys.modules[name]
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch unavailable: {exc}")
        if not hasattr(getattr(torch, "nn", None), "Module"):
            self.skipTest("real torch.nn.Module unavailable")

        from blb_stage2_rl.sequential_policy import (
            SequentialPPOConfig,
            SequentialRolloutBuffer,
            sequential_ppo_update,
        )
        import numpy as np

        class SpyPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))
                self.seen_masks = []

            def evaluate_action(
                    self,
                    state,
                    actions,
                    slot_mask,
                    per_slot_num_levels,
                    action_level_mask=None,
                    baseline_prior_scale=None,
                    return_per_slot_entropy=False,
                    ):
                self.seen_masks.append(
                    None if action_level_mask is None
                    else action_level_mask.detach().cpu().clone()
                )
                batch = state.shape[0]
                base = self.weight.expand(batch)
                entropy_per_slot = torch.ones_like(slot_mask, dtype=torch.float32) * 0.5
                if return_per_slot_entropy:
                    return base, base + 0.5, base, entropy_per_slot
                return base, base + 0.5, base

        level_mask = np.array(
            [
                [True, False, False],
                [False, True, False],
            ],
            dtype=bool,
        )
        buffer = SequentialRolloutBuffer()
        buffer.add(
            state=np.array([0.0, 1.0], dtype=np.float32),
            action=np.array([0, 1], dtype=np.int64),
            slot_mask=np.array([True, True], dtype=bool),
            per_slot_num_levels=np.array([3, 3], dtype=np.int64),
            action_level_mask=level_mask,
            log_prob=0.0,
            value=0.0,
            reward=1.0,
            done=True,
            baseline_prior_scale=0.75,
        )
        policy = SpyPolicy()
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.0)
        sequential_ppo_update(
            policy,
            optimizer,
            buffer,
            SequentialPPOConfig(n_epochs=1, minibatch_size=1, ent_coef=0.0),
            torch.device("cpu"),
        )
        self.assertTrue(policy.seen_masks)
        self.assertTrue(all(mask is not None for mask in policy.seen_masks))
        self.assertTrue(torch.equal(policy.seen_masks[0][0], torch.from_numpy(level_mask)))

    def test_sequential_ppo_update_skips_nonfinite_minibatch(self):
        import math
        import sys

        torch_mod = sys.modules.get("torch")
        if torch_mod is not None and not hasattr(getattr(torch_mod, "nn", None), "Module"):
            for name in list(sys.modules):
                if name == "torch" or name.startswith("torch."):
                    del sys.modules[name]
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch unavailable: {exc}")
        if not hasattr(getattr(torch, "nn", None), "Module"):
            self.skipTest("real torch.nn.Module unavailable")

        from blb_stage2_rl.sequential_policy import (
            SequentialPPOConfig,
            SequentialRolloutBuffer,
            sequential_ppo_update,
        )
        import numpy as np

        class NonfinitePolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))
                self._ppo_lr_scale = 1.0
                self._ppo_last_avg_kl = 0.0

            def evaluate_action(
                    self,
                    state,
                    actions,
                    slot_mask,
                    per_slot_num_levels,
                    action_level_mask=None,
                    baseline_prior_scale=None,
                    return_per_slot_entropy=False,
                    ):
                batch = state.shape[0]
                bad = self.weight.expand(batch) * torch.tensor(float("nan"))
                entropy_per_slot = torch.ones_like(slot_mask, dtype=torch.float32)
                if return_per_slot_entropy:
                    return bad, bad, bad, entropy_per_slot
                return bad, bad, bad

        buffer = SequentialRolloutBuffer()
        buffer.add(
            state=np.array([0.0, 1.0], dtype=np.float32),
            action=np.array([0, 1], dtype=np.int64),
            slot_mask=np.array([True, True], dtype=bool),
            per_slot_num_levels=np.array([3, 3], dtype=np.int64),
            action_level_mask=np.ones((2, 3), dtype=bool),
            log_prob=0.0,
            value=0.0,
            reward=1.0,
            done=True,
            baseline_prior_scale=0.75,
        )
        policy = NonfinitePolicy()
        optimizer = torch.optim.SGD(policy.parameters(), lr=1.0)
        metrics = sequential_ppo_update(
            policy,
            optimizer,
            buffer,
            SequentialPPOConfig(
                n_epochs=1,
                minibatch_size=1,
                ent_coef=0.0,
                kl_adaptive_max_ratio=1.25,
            ),
            torch.device("cpu"),
        )
        self.assertGreater(metrics["nonfinite_minibatches"], 0)
        self.assertTrue(metrics["nonfinite_update_skipped"])
        self.assertTrue(math.isfinite(metrics["policy_loss"]))
        self.assertTrue(math.isfinite(metrics["value_loss"]))
        self.assertTrue(torch.isfinite(policy.weight).item())
        self.assertLessEqual(policy._ppo_lr_scale, 1.0)

    def test_sequential_ppo_update_handles_budgeted_reward_extremes(self):
        import math
        import sys

        torch_mod = sys.modules.get("torch")
        if torch_mod is not None and not hasattr(getattr(torch_mod, "nn", None), "Module"):
            for name in list(sys.modules):
                if name == "torch" or name.startswith("torch."):
                    del sys.modules[name]
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch unavailable: {exc}")
        if not hasattr(getattr(torch, "nn", None), "Module"):
            self.skipTest("real torch.nn.Module unavailable")

        from blb_stage2_rl.sequential_policy import (
            SequentialPPOConfig,
            SequentialRolloutBuffer,
            sequential_ppo_update,
        )
        import numpy as np

        class FinitePolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))
                self._ppo_lr_scale = 1.0
                self._ppo_last_avg_kl = 0.0

            def evaluate_action(
                    self,
                    state,
                    actions,
                    slot_mask,
                    per_slot_num_levels,
                    action_level_mask=None,
                    baseline_prior_scale=None,
                    return_per_slot_entropy=False,
                    ):
                batch = state.shape[0]
                logp = self.weight.expand(batch)
                value = (self.weight * 0.5).expand(batch)
                entropy_per_slot = torch.ones_like(slot_mask, dtype=torch.float32) * 0.5
                entropy = entropy_per_slot.sum(dim=-1)
                if return_per_slot_entropy:
                    return logp, entropy, value, entropy_per_slot
                return logp, entropy, value

        buffer = SequentialRolloutBuffer()
        for reward_value in (-5.0, 20.0, 40.0, 45.0):
            buffer.add(
                state=np.array([0.0, 1.0], dtype=np.float32),
                action=np.array([0, 1], dtype=np.int64),
                slot_mask=np.array([True, True], dtype=bool),
                per_slot_num_levels=np.array([3, 3], dtype=np.int64),
                action_level_mask=np.ones((2, 3), dtype=bool),
                log_prob=0.0,
                value=0.0,
                reward=float(reward_value),
                done=True,
                baseline_prior_scale=0.75,
            )
        policy = FinitePolicy()
        optimizer = torch.optim.SGD(policy.parameters(), lr=1.0e-3)
        metrics = sequential_ppo_update(
            policy,
            optimizer,
            buffer,
            SequentialPPOConfig(
                n_epochs=1,
                minibatch_size=2,
                ent_coef=0.0,
                kl_adaptive_max_ratio=1.25,
            ),
            torch.device("cpu"),
        )
        for key in ("policy_loss", "value_loss", "approx_kl", "lr_scale"):
            self.assertTrue(math.isfinite(float(metrics[key])), key)
        self.assertEqual(int(metrics.get("nonfinite_minibatches", 0)), 0)
        self.assertTrue(torch.isfinite(policy.weight).item())

    def test_sequential_ppo_adaptive_lr_cap_is_conservative_for_gtrxl(self):
        from pathlib import Path

        src = Path("blb_stage2_rl/sequential_policy.py").read_text(encoding="utf-8")
        self.assertIn("kl_adaptive_max_ratio: float = 1.25", src)

    def test_sequential_runner_uses_episode_neighbor_offsets(self):
        from pathlib import Path

        src = Path("blb_stage2_rl/sequential_runner.py").read_text(encoding="utf-8")
        for needle in (
            "_sample_episode_neighbor_offsets",
            "neighbor_selected_offsets",
            "_build_step_level_mask",
            "warmstart_neighbor_sampling",
        ):
            self.assertIn(needle, src, msg=f"missing safe curriculum wiring: {needle!r}")

    def test_new_best_logs_inference_metrics(self):
        """After a new best, the log line should include loss_mean / loss_std /
        m1 so the user can verify acc/stab gates without grepping details files.
        """
        src = open("blb_stage2_rl/sequential_runner.py", encoding="utf-8").read()
        self.assertIn("推理指标（inference test metrics）", src)
        self.assertIn("record.terminal_loss_mean", src)
        self.assertIn("record.terminal_loss_std", src)
        self.assertIn("record.terminal_metric1_mean", src)
        self.assertIn("record.terminal_metric2_mean", src)
        self.assertIn("record.terminal_metric1_std", src)
        self.assertIn("record.terminal_metric2_std", src)
        self.assertIn("record.terminal_stab_violation", src)


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
        a 600-episode ramp."""
        src = open("blb_stage2_rl/sequential_runner.py", encoding="utf-8").read()
        self.assertIn("ent_coef_anchor: float = 0.0", src)
        self.assertIn("ent_coef_ramp_episodes: int = 600", src)
        # BLBStage2TrainConfig (used by the runner) must also expose them
        # so the launcher / preset can override.
        src2 = open("blb_stage2_rl/runner.py", encoding="utf-8").read()
        self.assertIn("ent_coef_anchor: float = 0.0", src2)
        self.assertIn("ent_coef_ramp_episodes: int = 600", src2)

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
        # ep=360 = 60 anchor + 300 into ramp; ramp_episodes default 600 -> 50% ramp
        self.assertAlmostEqual(
            fn(ep_count_1based=360, anchor_episodes=60, target_ent_coef=0.02),
            0.01, places=5,
        )

        # End of ramp: target
        self.assertAlmostEqual(
            fn(ep_count_1based=660, anchor_episodes=60, target_ent_coef=0.02),
            0.02, places=5,
        )

        # Steady: target
        for ep in (661, 1000, 6000):
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
