from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]

PRESERVED_RESULT_ROOTS = (
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
)

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
}

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


def is_preserved_result(path: str) -> bool:
    return path in {
        "glue_final_configs_best_genetic.json",
        "glue_final_configs_best_ppo.json",
    } or path.startswith(PRESERVED_RESULT_ROOTS)


def active_source_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        path
        for path in paths
        if not is_preserved_result(path)
        and path != "scripts/production_surface_guard.py"
        and not path.startswith(("tests/", "docs/", "agent_handoffs/"))
        and Path(path).suffix in ACTIVE_SUFFIXES
    )


def tracked_text(relative: str) -> str:
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
        self.assertEqual(sorted(paths & FORBIDDEN_RUNTIME_PATHS), [])

    def test_backup_source_files_are_absent(self):
        offenders = [
            path
            for path in tracked_paths()
            if not is_preserved_result(path)
            and (".bak" in path or "legacy_results" in path)
        ]
        self.assertEqual(sorted(offenders), [])

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
