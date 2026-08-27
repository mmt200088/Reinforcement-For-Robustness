#!/usr/bin/env python3
"""Fail when retired runtime paths or selectors re-enter tracked source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Iterable


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
    "outputs/",
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
    "fixtures/reproducibility/mrpc_validation_v1.json",
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
    "outputs/README.md",
}

FORBIDDEN_RUNTIME_PATHS = {
    "llama_7B_LayerImportance.sh",
    "rl_tune.py",
    "layer_importance_evaluator.py",
    "function_handler.py",
    "blb_rl_bridge.py",
    "device_utils.py",
    "elastic_gpu.py",
    "noise_tables.py",
    "noise_targets_registry.py",
    "src/rfr/evaluation/action_grid.py",
    "src/rfr/evaluation/embedded.py",
    "src/rfr/evaluation/final_eval_layout.py",
    "src/rfr/evaluation/final_evaluation.py",
    "src/rfr/preparation/data/mrpc_reproducibility.py",
    "rl_data_points.py",
    "rl_local_optimum.py",
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
    "Model_analysis/",
    "Paean/",
    "Rescale_optimizer/",
    "blb_stage2_rl/",
    "config/",
    "presets/",
    "Rescale_optimizer/replan_configs/",
    "stage1_rl/",
    "tools/",
)

REQUIRED_RUNTIME_PATHS = {
    "run_search.sh",
    "src/rfr/cli/run.py",
    "src/rfr/cli/evaluate.py",
    "src/rfr/evaluation/protocol.py",
    "src/rfr/preparation/data/protocol.py",
    "src/rfr/preparation/data/mrpc_contract.py",
    "src/rfr/preparation/fusion/count_map.py",
    "src/rfr/preparation/rescale/optimizer/replan_interface.py",
    "src/rfr/search/rl/stage1/parallel_runner.py",
    "src/rfr/search/rl/stage2/training.py",
    "src/rfr/search/common/best_config.py",
    "src/rfr/search/comparators/bo_rf/stage1.py",
    "src/rfr/search/comparators/bo_rf/stage2.py",
    "src/rfr/search/comparators/greedy/stage1.py",
    "src/rfr/search/comparators/greedy/stage2.py",
    "src/rfr/search/comparators/coinn_ga/stage1.py",
    "src/rfr/search/comparators/coinn_ga/stage2.py",
    "src/rfr/evaluation/action_eval.py",
    "outputs/README.md",
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
    "--comparator-smoke",
    "comparator_smoke",
    "final_eval_config_source",
    "manual_stage1_gelu",
    "manual_stage2_noise",
    "final_eval_action_ranges",
    "final_eval_action_fixed",
    "final_eval_cost_match_count",
    "evaluation_cap",
    "search_evaluation_budget",
    "ga_stop_on_no_improvement",
    "ga_require_full_generations",
    "convergence_patience_updates",
    "stage1_entropy_stop_threshold",
    "smoke_only_complete",
    "two_stage_result.json",
    "mrpc_validation_v1.json",
    "canonical_rows",
    "evaluate_model_with_attention_noise",
    "replace_layer_input_noise",
    "candidate_trial_group_v1",
    "candidate_promotion_status_v1",
    "allow_legacy",
    "_COMPAT_EXTRA_FIELDS",
    "select_fusion_eval_metadata",
    "best_noise_config",
    "stable_search_best_noise_config",
    "ActionRandom_",
    "BLBActionRandom",
)

ACTIVE_SUFFIXES = {".py", ".sh", ".json", ".toml", ".conf"}
NON_RUNTIME_SOURCE_ROOTS = (
    "agent_handoffs/",
    "docs/",
    "examples/",
    "local_assets/",
)


def _git(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout


def tracked_paths() -> tuple[str, ...]:
    return tuple(
        value.decode("utf-8")
        for value in _git("ls-files", "-z").split(b"\0")
        if value
    )


def is_allowed_result(path: str) -> bool:
    return path in ALLOWED_RESULT_FILES


def active_source_paths(paths: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        path
        for path in paths
        if not is_allowed_result(path)
        and path != "scripts/production_surface_guard.py"
        and not path.startswith(NON_RUNTIME_SOURCE_ROOTS)
        and Path(path).suffix in ACTIVE_SUFFIXES
    )


def tracked_text(path: str) -> str:
    worktree_path = REPO_ROOT / path
    if worktree_path.is_file():
        return worktree_path.read_text(encoding="utf-8", errors="replace")
    return _git("show", f"HEAD:{path}").decode("utf-8", errors="replace")


def audit() -> dict[str, object]:
    paths = tracked_paths()
    path_set = set(paths)
    missing_required_paths = sorted(REQUIRED_RUNTIME_PATHS - path_set)
    forbidden_paths = sorted(path_set & FORBIDDEN_RUNTIME_PATHS)
    forbidden_paths.extend(sorted(
        path
        for path in paths
        if any(path.startswith(prefix) for prefix in FORBIDDEN_RUNTIME_PREFIXES)
    ))
    tracked_artifact_paths = sorted(
        path
        for path in paths
        if not is_allowed_result(path)
        and (
            path in FORBIDDEN_TRACKED_ARTIFACT_FILES
            or any(path.startswith(prefix) for prefix in FORBIDDEN_TRACKED_ARTIFACT_ROOTS)
            or Path(path).suffix.lower() in FORBIDDEN_WEIGHT_SUFFIXES
        )
    )
    backup_paths = sorted(
        path
        for path in paths
        if not is_allowed_result(path)
        and (".bak" in path or "legacy_results" in path)
    )
    reference_hits: list[dict[str, str]] = []
    for path in active_source_paths(paths):
        source = tracked_text(path)
        for token in FORBIDDEN_RUNTIME_REFERENCES:
            if token in source:
                reference_hits.append({"path": path, "token": token})
    return {
        "ok": (
            not missing_required_paths
            and not forbidden_paths
            and not tracked_artifact_paths
            and not backup_paths
            and not reference_hits
        ),
        "missing_required_paths": missing_required_paths,
        "forbidden_paths": forbidden_paths,
        "tracked_artifact_paths": tracked_artifact_paths,
        "backup_paths": backup_paths,
        "reference_hits": reference_hits,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for key in (
            "missing_required_paths",
            "forbidden_paths",
            "tracked_artifact_paths",
            "backup_paths",
            "reference_hits",
        ):
            values = result[key]
            if values:
                print(f"{key}:")
                for value in values:
                    print(f"  {value}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
