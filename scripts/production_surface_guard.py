#!/usr/bin/env python3
"""Fail when retired runtime paths or selectors re-enter tracked source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Iterable


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

PRESERVED_RESULT_FILES = {
    "glue_final_configs_best_genetic.json",
    "glue_final_configs_best_ppo.json",
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
NON_RUNTIME_SOURCE_ROOTS = ("tests/", "docs/", "agent_handoffs/")


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


def is_preserved_result(path: str) -> bool:
    return path in PRESERVED_RESULT_FILES or path.startswith(
        PRESERVED_RESULT_ROOTS
    )


def active_source_paths(paths: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        path
        for path in paths
        if not is_preserved_result(path)
        and path != "scripts/production_surface_guard.py"
        and not path.startswith(NON_RUNTIME_SOURCE_ROOTS)
        and Path(path).suffix in ACTIVE_SUFFIXES
    )


def tracked_text(path: str) -> str:
    return _git("show", f"HEAD:{path}").decode("utf-8", errors="replace")


def audit() -> dict[str, object]:
    paths = tracked_paths()
    path_set = set(paths)
    forbidden_paths = sorted(path_set & FORBIDDEN_RUNTIME_PATHS)
    backup_paths = sorted(
        path
        for path in paths
        if not is_preserved_result(path)
        and (".bak" in path or "legacy_results" in path)
    )
    reference_hits: list[dict[str, str]] = []
    for path in active_source_paths(paths):
        source = tracked_text(path)
        for token in FORBIDDEN_RUNTIME_REFERENCES:
            if token in source:
                reference_hits.append({"path": path, "token": token})
    return {
        "ok": not forbidden_paths and not backup_paths and not reference_hits,
        "forbidden_paths": forbidden_paths,
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
        for key in ("forbidden_paths", "backup_paths", "reference_hits"):
            values = result[key]
            if values:
                print(f"{key}:")
                for value in values:
                    print(f"  {value}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
