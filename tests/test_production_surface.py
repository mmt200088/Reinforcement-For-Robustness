from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_TRACKED_ARTIFACT_ROOTS = (
    ".claude/",
    ".vscode/",
    "server_backups/",
    "rl_training_data_points/",
    "Parting Chapter/",
    "Prelude Chapter/",
    "Previous Chapter/",
    "Previous Chapter Server Reserve/",
    "Paean/outputs/",
    "experiment/outputs/",
    "experiments/",
    "gelu_analysis/",
    "glue_submission/",
    "reports/",
    "Model_analysis/model_statistics/weight_hist_out/",
    "Rescale_optimizer/diagnose_certacc_output/",
    "docs/assets/",
    "docs/evidence/",
)

FORBIDDEN_TRACKED_ARTIFACT_FILES = {
    "commonsense_170k.json",
    "noise_std_table.csv",
    "pruning_search_log_eval.txt",
    "rl_agent_checkpoint_BertForSequenceClassification.pt",
}

FORBIDDEN_WEIGHT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
}

ALLOWED_RESULT_FILES = {
    "examples/representative_rl_log/README.md",
    "examples/representative_rl_log/stage2_mrpc_600ep.jsonl",
    "configs/reference/coinn_ga.json",
    "configs/reference/rl.json",
}

FORBIDDEN_RUNTIME_PATHS = {
    "noise_rl_module_v2.py",
    "general_policy_module.py",
    "rl_tune_general.py",
    "rl_ga_compare_runner.py",
    "genetic_search_module.py",
    "greedy_search_module.py",
    "grpo_common.py",
    "blb_stage2_rl/network_variants.py",
    "blb_stage2_rl/action_mask.py",
    "blb_stage2_rl/policy.py",
    "blb_stage2_rl/parallel_runner.py",
    "blb_stage2_rl/sequential_env.py",
    "blb_stage2_rl/substage_env.py",
    "blb_stage2_rl/substage_runner.py",
    "blb_stage2_rl/osr.py",
    "blb_stage2_rl/fusion_curriculum.py",
    "blb_stage2_rl/protected_k1.py",
    "blb_stage2_rl/same_action_parity.py",
    "approximation.py",
    "approximation_exp.py",
    "bert-test.py",
    "commonsense_evaluate.py",
    "moe_sample.py",
    "distribution_stats.py",
    "scripts/blb_phase0_preflight.py",
    "Rescale_optimizer/scripts/check_compress_headroom.py",
    "Rescale_optimizer/scripts/diagnose_certacc.py",
    "Rescale_optimizer/scripts/replan_what_if.py",
    "Rescale_optimizer/scripts/sweep_certacc_monotonicity.py",
    "Rescale_optimizer/scripts/update_noise_tables_from_csv.py",
    "Model_analysis/model_statistics/weight_hist_magnitude_nosample.py",
    "Rescale_optimizer/rescale_optimizer/accuracy_preservation.md",
    "Rescale_optimizer/rescale_optimizer/accuracy_preservation_render.md",
    "Rescale_optimizer/rescale_optimizer/accuracy_preservation_v1_subgaussian.md",
    "Rescale_optimizer/rescale_optimizer/algorithms.md",
    "Rescale_optimizer/rescale_optimizer/example.py",
}

FORBIDDEN_RUNTIME_PREFIXES = (
    "Rescale_optimizer/replan_configs/",
)

FORBIDDEN_RUNTIME_REFERENCES = (
    "legacy_v2",
    "separate_critic_gtrxl_v1",
    "separate_critic_mlp_v1",
    "stage2_rl_devices",
    "substage_mode",
    "osr_scan_only",
    "rescale_invoker_kind",
    "HeuristicStubInvoker",
    "SubprocessInvoker",
    "importance-aware-sparse-tuning-IST-paper",
    "class PolicyNetwork(",
    "class ValueNetwork(",
    "class DualHeadValueNetwork(",
    "class LSTMStrategyNetwork(",
    "DisentangledNormalizer",
    "use_ist",
    "adapter_name",
    "lora_target_modules",
    "wandb_project",
    "transformers.models.gpt2",
    "_GPT2_PATHS",
    "_gpt2_qkv",
    "block5_n0",
    "STAGE1_ENABLE_DIFFERENTIAL_REWARD",
    "RL_OPT_FLAGS",
    "BLB_TRUNCATION_K_LEVELS",
    "block1_wnli",
    "block2_wnli",
    "includes_first_input",
    "replace_blb_first_input_noise",
    "restore_blb_first_input_noise",
    "_make_blb_first_input_noise_forward",
    "blb_first_input_noise_state",
    "_run_legacy_preflight_if_needed",
    "<!doctype html>",
    "html_table(",
    "pareto_html_path",
)

ACTIVE_SUFFIXES = {".py", ".sh", ".json", ".toml", ".conf"}


def tracked_paths() -> tuple[str, ...]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return tuple(
        value.decode("utf-8")
        for value in completed.stdout.split(b"\0")
        if value
    )


def is_allowed_result(path: str) -> bool:
    return path in ALLOWED_RESULT_FILES


def active_source_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        path
        for path in paths
        if not is_allowed_result(path)
        and path != "scripts/production_surface_guard.py"
        and not path.startswith(("tests/", "docs/", "agent_handoffs/"))
        and Path(path).suffix in ACTIVE_SUFFIXES
    )


def tracked_text(relative: str) -> str:
    worktree_path = REPO_ROOT / relative
    if worktree_path.is_file():
        return worktree_path.read_text(encoding="utf-8", errors="replace")
    completed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return completed.stdout.decode("utf-8", errors="replace")


class ProductionSurfaceTests(unittest.TestCase):
    def test_guard_cli_module_exists(self):
        self.assertIsNotNone(
            importlib.util.find_spec("scripts.production_surface_guard")
        )

    def test_obsolete_runtime_paths_are_absent(self):
        paths = set(tracked_paths())
        offenders = set(paths & FORBIDDEN_RUNTIME_PATHS)
        offenders.update(
            path
            for path in paths
            if any(path.startswith(prefix) for prefix in FORBIDDEN_RUNTIME_PREFIXES)
        )
        self.assertEqual(sorted(offenders), [])

    def test_backup_source_files_are_absent(self):
        offenders = [
            path
            for path in tracked_paths()
            if not is_allowed_result(path)
            and (".bak" in path or "legacy_results" in path)
        ]
        self.assertEqual(sorted(offenders), [])

    def test_generated_artifacts_are_not_tracked(self):
        offenders = [
            path
            for path in tracked_paths()
            if path in FORBIDDEN_TRACKED_ARTIFACT_FILES
            or any(path.startswith(prefix) for prefix in FORBIDDEN_TRACKED_ARTIFACT_ROOTS)
            or Path(path).suffix.lower() in FORBIDDEN_WEIGHT_SUFFIXES
        ]
        self.assertEqual(sorted(offenders), [])

    def test_required_compact_results_are_tracked(self):
        paths = set(tracked_paths())
        self.assertTrue(ALLOWED_RESULT_FILES.issubset(paths))

    def test_local_noise_table_is_ignored(self):
        completed = subprocess.run(
            ["git", "check-ignore", "--quiet", "noise_std_table.csv"],
            cwd=REPO_ROOT,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)

    def test_representative_log_is_bounded(self):
        log_path = REPO_ROOT / "examples/representative_rl_log/stage2_mrpc_600ep.jsonl"
        self.assertGreaterEqual(log_path.stat().st_size, 5 * 1024 * 1024)
        self.assertLessEqual(log_path.stat().st_size, 10 * 1024 * 1024)
        with log_path.open("rb") as handle:
            self.assertEqual(sum(1 for _ in handle), 600)

    def test_obsolete_runtime_references_are_absent(self):
        offenders: list[tuple[str, str]] = []
        paths = tracked_paths()
        for relative in active_source_paths(paths):
            text = tracked_text(relative)
            for token in FORBIDDEN_RUNTIME_REFERENCES:
                if token in text:
                    offenders.append((relative, token))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
