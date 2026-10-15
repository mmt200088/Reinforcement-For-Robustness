#!/usr/bin/env python3
"""Fail when retired runtime paths or selectors re-enter tracked source."""

from __future__ import annotations

import argparse
import ast
import importlib.util
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
    "src/rfr/search/common/evaluator.py",
    "src/rfr/search/common/action_space.py",
    "src/rfr/evaluation/training_curve_plot.py",
    "src/rfr/search/rl/stage2/training.py",
    "src/rfr/search/rl/stage2/runner.py",
    "src/rfr/search/rl/stage2/layerwise_runner.py",
    "src/rfr/search/rl/stage2/env.py",
    "src/rfr/search/rl/stage2/layerwise_env.py",
    "src/rfr/search/runtime/blb_bridge.py",
    "src/rfr/preparation/rescale/bridge.py",
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
    "src/rfr/search/rl/stage2/entrypoint.py",
    "src/rfr/search/rl/stage2/orchestration.py",
    "src/rfr/search/rl/stage2/ppo_runner.py",
    "src/rfr/search/rl/stage2/policy_env.py",
    "src/rfr/search/rl/stage2/execution_env.py",
    "src/rfr/search/workflow.py",
    "src/rfr/search/rl/stage1/config.py",
    "src/rfr/search/rl/stage1/policy.py",
    "src/rfr/search/rl/stage1/env.py",
    "src/rfr/search/rl/stage1/rollout.py",
    "src/rfr/search/rl/stage1/training.py",
    "src/rfr/search/rl/stage1/runner.py",
    "src/rfr/search/common/full_action.py",
    "src/rfr/search/common/run_artifacts.py",
    "src/rfr/search/common/training_curves.py",
    "src/rfr/search/runtime/model_installation.py",
    "src/rfr/preparation/rescale/optimizer_bridge.py",
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


def module_structure_errors(paths: Iterable[str]) -> list[str]:
    """Check local imports and algorithm ownership without importing Torch."""
    modules: dict[str, ast.Module] = {}
    packages: set[str] = set()
    errors: list[str] = []
    for path in paths:
        if not path.startswith("src/rfr/") or not path.endswith(".py"):
            continue
        module = path[4:-3].replace("/", ".")
        if module.endswith(".__init__"):
            module = module.removesuffix(".__init__")
            packages.add(module)
        try:
            modules[module] = ast.parse(tracked_text(path), filename=path)
        except SyntaxError as exc:
            errors.append(f"{path}: {exc}")

    exports: dict[str, set[str]] = {}
    for module, tree in modules.items():
        names: set[str] = set()
        pending = list(tree.body)
        while pending:
            node = pending.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names.update(target.id for target in targets if isinstance(target, ast.Name))
                if any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        names.update(
                            item.value for item in node.value.elts
                            if isinstance(item, ast.Constant) and isinstance(item.value, str)
                        )
            else:
                # Module-level conditional/optional exports, not function locals.
                pending.extend(ast.iter_child_nodes(node))
        exports[module] = names

    for module, tree in modules.items():
        package = module if module in packages else module.rsplit(".", 1)[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                target = node.module or ""
                if node.level:
                    target = importlib.util.resolve_name("." * node.level + target, package)
                targets = [target]
                if target in modules:
                    for alias in node.names:
                        if alias.name != "*" and alias.name not in exports[target] and f"{target}.{alias.name}" not in modules:
                            errors.append(f"{module}:{node.lineno}: missing local import {target}.{alias.name}")
                if target == "rfr.search.rl.stage1.config":
                    if any(alias.name in {"PPO_UPDATE_INTERVAL", "PPO_BATCH_SIZE", "STEP_INFO_CHUNK_SIZE"} for alias in node.names):
                        errors.append(f"{module}:{node.lineno}: mutable rollout settings must be read through the config module")
            else:
                continue
            for target in targets:
                if target.startswith("rfr.") and target not in modules:
                    errors.append(f"{module}:{node.lineno}: missing local module {target}")
                if target == "rfr.search.workflow" and module.startswith(("rfr.search.common.", "rfr.search.rl.stage1.")):
                    errors.append(f"{module}:{node.lineno}: lower-level code must not import the workflow")

    owners = {
        "SearchWorkflow": "rfr.search.workflow",
        "GTrXLStrategyNetwork": "rfr.search.rl.stage1.policy",
        "TransformerOptEnv": "rfr.search.rl.stage1.env",
        "RecurrentRolloutBuffer": "rfr.search.rl.stage1.rollout",
        "ppo_update_gtrxl": "rfr.search.rl.stage1.training",
        "run_stage1": "rfr.search.rl.stage1.runner",
    }
    for name, expected in owners.items():
        actual = sorted(
            module for module, tree in modules.items()
            if any(isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name for node in ast.walk(tree))
        )
        if actual != [expected]:
            errors.append(f"{name}: expected sole owner {expected}, got {actual}")
    return sorted(errors)


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
    structure_errors = module_structure_errors(paths)
    return {
        "ok": (
            not missing_required_paths
            and not forbidden_paths
            and not tracked_artifact_paths
            and not backup_paths
            and not reference_hits
            and not structure_errors
        ),
        "missing_required_paths": missing_required_paths,
        "forbidden_paths": forbidden_paths,
        "tracked_artifact_paths": tracked_artifact_paths,
        "backup_paths": backup_paths,
        "reference_hits": reference_hits,
        "module_structure_errors": structure_errors,
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
            "module_structure_errors",
        ):
            values = result[key]
            if values:
                print(f"{key}:")
                for value in values:
                    print(f"  {value}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
