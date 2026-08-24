from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path
import re
import shlex
import time
from typing import List, Optional, Sequence, Tuple

from glue_data_protocol import (
    SUPPORTED_DATASETS,
    SUPPORTED_MODEL_FAMILIES,
    validate_supported_profile,
)
from runtime_error_reporter import format_command

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
PRESET_DIR = PACKAGE_DIR / "presets"
DEFAULT_OUTPUT_ROOT = PACKAGE_DIR / "outputs"
DEFAULT_PRESET = "default"

DATASET_CHOICES = SUPPORTED_DATASETS
MODEL_TYPE_CHOICES = SUPPORTED_MODEL_FAMILIES
SOURCE_CHOICES = ("search", "json", "manual", "max", "stage2-max", "stage2_max", "blb-max", "blb_max")


@dataclasses.dataclass(frozen=True)
class FinalEvalSettings:
    preset: str = ""
    dataset: str = "mrpc"
    algorithm: str = "rl"
    model_type: str = "bert-base"
    batch_size: int = 16
    logfile: str = "final_eval.log"
    source: str = "json"
    config: str = "glue_final_configs_best_ppo.json"
    resume_from: str = ""
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    run_name: str = ""
    repeat: int = 50
    random_seed: int = 42
    perm_trials: int = 0
    cost_trials: int = 0
    budget_trials: int = 0
    stage1_budget_trials: int = 0
    stage2_budget_trials: int = 0
    random_enabled: bool = False
    action_config: str = ""
    action_ranges: Tuple[str, ...] = dataclasses.field(default_factory=tuple)
    action_fixed: Tuple[str, ...] = dataclasses.field(default_factory=tuple)
    blb_rescale_invoker_kind: str = "heuristic"
    blb_rescale_optimizer_root: str = ""
    require_rescale_optimizer: bool = False
    stage1_accuracy_tolerance: float = 0.005
    stage2_limit_tolerance: float = 0.05
    stage2_stability_tolerance: float = 0.05
    stage2_k_trials: int = 5
    stage2_probe_size: int = 256
    stage2_rl_variant: str = "blb_v3"
    manual_stage1_gelu: str = ""
    manual_stage1_softmax: str = ""
    manual_stage2_noise: str = ""
    cost_match_count: int = 50
    cost_match_max_attempts: int = 5000
    glue_submission_enabled: bool = True
    glue_submission_seed: int = 42
    dry_run: bool = False
    foreground: bool = False


def _read_preset_args(preset_name: str, preset_dir: Path = PRESET_DIR) -> List[str]:
    name = str(preset_name or "").strip()
    if not name:
        return []
    path = preset_dir / f"{name}.conf"
    if not path.is_file():
        available = " ".join(list_presets(preset_dir))
        raise FileNotFoundError(
            f"final_eval preset not found: {path}. Available presets: {available or '(none)'}"
    )

    args: List[str] = []
    with path.open(encoding="utf-8-sig") as handle:
        for raw_line in handle:
            lexer = shlex.shlex(raw_line, posix=True)
            lexer.whitespace_split = True
            lexer.commenters = "#"
            args.extend(list(lexer))
    return args


def list_presets(preset_dir: Path = PRESET_DIR) -> List[str]:
    if not preset_dir.is_dir():
        return []
    try:
        with os.scandir(preset_dir) as entries:
            return sorted(
                entry.name[: -len(".conf")]
                for entry in entries
                if entry.name.endswith(".conf") and entry.is_file()
            )
    except OSError:
        return []


def expand_preset_args(argv: Sequence[str], preset_dir: Path = PRESET_DIR) -> List[str]:
    raw = list(argv)
    expanded: List[str] = []
    remaining: List[str] = []
    i = 0
    while i < len(raw):
        token = raw[i]
        if token == "--preset":
            if i + 1 >= len(raw):
                raise ValueError("--preset requires a value")
            preset_name = raw[i + 1]
            expanded.extend(_read_preset_args(preset_name, preset_dir=preset_dir))
            remaining.extend(["--preset", preset_name])
            i += 2
            continue
        if token.startswith("--preset="):
            preset_name = token.split("=", 1)[1]
            expanded.extend(_read_preset_args(preset_name, preset_dir=preset_dir))
            remaining.append(token)
            i += 1
            continue
        remaining.append(token)
        i += 1
    return expanded + remaining


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Paean/run_final_eval.sh",
        description="Run the independent unified final-eval module.",
    )
    parser.add_argument("--preset", default="")
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--dataset", default="mrpc", choices=DATASET_CHOICES)
    parser.add_argument("--algorithm", "--search-algorithm", default="rl")
    parser.add_argument(
        "--model-type", default="bert-base", choices=MODEL_TYPE_CHOICES
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--logfile", default="final_eval.log")
    parser.add_argument("--source", "--config-source", "--final-eval-source", dest="source", default="json")
    parser.add_argument("--config", "--final-eval-config", dest="config", default="")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--repeat", "--eval-repeat", "--final-eval-repeat", dest="repeat", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--random", "--enable-random", dest="random_enabled", action="store_true")
    parser.add_argument("--perm-trials", "--permutation-trials", dest="perm_trials", type=int, default=0)
    parser.add_argument("--cost-trials", dest="cost_trials", type=int, default=0)
    parser.add_argument("--budget-trials", dest="budget_trials", type=int, default=0)
    parser.add_argument("--budget", dest="simple_budget", type=int, default=None)
    parser.add_argument("--stage1-budget-trials", dest="stage1_budget_trials", type=int, default=0)
    parser.add_argument("--stage2-budget-trials", dest="stage2_budget_trials", type=int, default=0)
    parser.add_argument("--stage1-accuracy-tolerance", type=float, default=0.005)
    parser.add_argument("--stage2-limit-tolerance", type=float, default=0.05)
    parser.add_argument("--stage2-stability-tolerance", type=float, default=0.05)
    parser.add_argument("--stage2-k-trials", type=int, default=5)
    parser.add_argument("--stage2-probe-size", type=int, default=256)
    parser.add_argument("--stage2-rl-variant", default="blb_v3")
    parser.add_argument("--manual-stage1-gelu", default="")
    parser.add_argument("--manual-stage1-softmax", default="")
    parser.add_argument("--manual-stage2-noise", default="")
    parser.add_argument("--action-config", default="")
    parser.add_argument("--action-range", "--range", dest="action_ranges", action="append", default=[])
    parser.add_argument("--action-fixed", "--fixed-action", dest="action_fixed", action="append", default=[])
    parser.add_argument("--rescale-invoker-kind", "--blb-rescale-invoker-kind", dest="blb_rescale_invoker_kind", default="heuristic")
    parser.add_argument("--rescale-optimizer-root", "--blb-rescale-optimizer-root", dest="blb_rescale_optimizer_root", default="")
    parser.add_argument("--require-rescale-optimizer", dest="require_rescale_optimizer", action="store_true")
    parser.add_argument(
        "--cost-match-count",
        dest="cost_match_count",
        type=int,
        default=50,
        help=(
            "Number of cost-matched random BLB action configs to draw and evaluate "
            "alongside the selected action (default 50). Set 0 to disable the "
            "same-cost comparison group even when --random is on."
        ),
    )
    parser.add_argument(
        "--cost-match-max-attempts",
        dest="cost_match_max_attempts",
        type=int,
        default=5000,
        help=(
            "Maximum number of random draws (incl. invalid + cost-mismatch) before "
            "the sampler stops trying to reach --cost-match-count. Default 5000."
        ),
    )
    parser.add_argument(
        "--glue-submission",
        dest="glue_submission_enabled",
        action="store_true",
        default=True,
        help="Generate a GLUE submission zip for the selected BLB action after final eval (default on).",
    )
    parser.add_argument(
        "--no-glue-submission",
        dest="glue_submission_enabled",
        action="store_false",
        help="Disable the post-final-eval GLUE submission generation.",
    )
    parser.add_argument(
        "--glue-submission-seed",
        dest="glue_submission_seed",
        type=int,
        default=42,
        help="RNG seed used for reproducible BLB-noise sampling during GLUE inference (default 42).",
    )
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def normalize_algorithm(value: str) -> str:
    raw = str(value or "rl").strip().lower().replace("_", "-")
    if raw in ("rl", "ppo"):
        return "rl"
    if raw in ("ga", "genetic"):
        return "ga"
    if raw in ("greedy", "greedy-search"):
        return "greedy"
    raise ValueError(f"Unsupported final_eval algorithm: {value!r}")


def default_config_for_algorithm(algorithm: str) -> str:
    if algorithm == "ga":
        return "glue_final_configs_best_genetic.json"
    if algorithm == "greedy":
        return "glue_final_configs_best_greedy.json"
    return "glue_final_configs_best_ppo.json"


def resolve_repo_path(path_value: str) -> str:
    value = str(path_value or "").strip()
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path)


def parse_final_eval_settings(
    argv: Optional[Sequence[str]] = None,
    *,
    preset_dir: Path = PRESET_DIR,
    require_resume_for_search: bool = True,
) -> FinalEvalSettings:
    raw_argv = list(argv or [])
    expanded = expand_preset_args(raw_argv, preset_dir=preset_dir)
    parser = build_parser()
    ns = parser.parse_args(expanded)

    algorithm = normalize_algorithm(ns.algorithm)
    source = str(ns.source or "json").strip().lower()
    if source not in SOURCE_CHOICES:
        raise ValueError(
            f"Unsupported final_eval source: {ns.source!r}. "
            f"Use one of: {', '.join(SOURCE_CHOICES)}"
        )
    config = ns.config or default_config_for_algorithm(algorithm)
    config = resolve_repo_path(config)
    resume_from = resolve_repo_path(ns.resume_from)
    output_root = resolve_repo_path(ns.output_root) or str(DEFAULT_OUTPUT_ROOT)

    perm_trials = ns.perm_trials
    cost_trials = ns.cost_trials
    budget_trials = ns.budget_trials
    stage1_budget_trials = ns.stage1_budget_trials
    stage2_budget_trials = ns.stage2_budget_trials
    if ns.simple_budget is not None:
        perm_trials = cost_trials = budget_trials = ns.simple_budget
        stage1_budget_trials = stage2_budget_trials = ns.simple_budget
    elif ns.random_enabled and not any(
        int(v) > 0
        for v in (
            perm_trials,
            cost_trials,
            budget_trials,
            stage1_budget_trials,
            stage2_budget_trials,
        )
    ):
        perm_trials = cost_trials = budget_trials = 10
        stage1_budget_trials = stage2_budget_trials = 10

    settings = FinalEvalSettings(
        preset=ns.preset,
        dataset=ns.dataset,
        algorithm=algorithm,
        model_type=str(ns.model_type or "bert-base").lower().replace("_", "-"),
        batch_size=int(ns.batch_size),
        logfile=ns.logfile,
        source=source,
        config=config,
        resume_from=resume_from,
        output_root=output_root,
        run_name=ns.run_name,
        repeat=int(ns.repeat),
        random_seed=int(ns.random_seed),
        perm_trials=int(perm_trials),
        cost_trials=int(cost_trials),
        budget_trials=int(budget_trials),
        stage1_budget_trials=int(stage1_budget_trials),
        stage2_budget_trials=int(stage2_budget_trials),
        random_enabled=bool(ns.random_enabled),
        action_config=resolve_repo_path(ns.action_config),
        action_ranges=tuple(str(v) for v in (ns.action_ranges or []) if str(v).strip()),
        action_fixed=tuple(str(v) for v in (ns.action_fixed or []) if str(v).strip()),
        blb_rescale_invoker_kind=str(ns.blb_rescale_invoker_kind or "heuristic").lower().replace("-", "_"),
        blb_rescale_optimizer_root=resolve_repo_path(ns.blb_rescale_optimizer_root),
        require_rescale_optimizer=bool(ns.require_rescale_optimizer),
        stage1_accuracy_tolerance=float(ns.stage1_accuracy_tolerance),
        stage2_limit_tolerance=float(ns.stage2_limit_tolerance),
        stage2_stability_tolerance=float(ns.stage2_stability_tolerance),
        stage2_k_trials=int(ns.stage2_k_trials),
        stage2_probe_size=int(ns.stage2_probe_size),
        stage2_rl_variant=str(ns.stage2_rl_variant or "blb_v3").lower(),
        manual_stage1_gelu=ns.manual_stage1_gelu,
        manual_stage1_softmax=ns.manual_stage1_softmax,
        manual_stage2_noise=ns.manual_stage2_noise,
        cost_match_count=int(ns.cost_match_count),
        cost_match_max_attempts=int(ns.cost_match_max_attempts),
        glue_submission_enabled=bool(ns.glue_submission_enabled),
        glue_submission_seed=int(ns.glue_submission_seed),
        dry_run=bool(ns.dry_run),
        foreground=bool(ns.foreground),
    )
    validate_settings(settings, require_resume_for_search=require_resume_for_search)
    return settings


def validate_settings(
    settings: FinalEvalSettings,
    *,
    require_resume_for_search: bool = True,
) -> None:
    if settings.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if settings.repeat <= 0:
        raise ValueError("--repeat/--eval-repeat must be positive")
    for name in (
        "perm_trials",
        "cost_trials",
        "budget_trials",
        "stage1_budget_trials",
        "stage2_budget_trials",
    ):
        if getattr(settings, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if not settings.random_enabled and any(
        getattr(settings, name) > 0
        for name in (
            "perm_trials",
            "cost_trials",
            "budget_trials",
            "stage1_budget_trials",
            "stage2_budget_trials",
        )
    ):
        raise ValueError(
            "Random comparison trial counts require --random/--enable-random. "
            "Without --random, standalone final_eval evaluates only the selected configs."
        )
    if not (0 < settings.stage1_accuracy_tolerance < 1):
        raise ValueError("--stage1-accuracy-tolerance must be a percentage in (0, 1)")
    if not (0 < settings.stage2_limit_tolerance < 1):
        raise ValueError("--stage2-limit-tolerance must be a percentage in (0, 1)")
    if not (0 < settings.stage2_stability_tolerance < 1):
        raise ValueError("--stage2-stability-tolerance must be a percentage in (0, 1)")
    if settings.stage2_k_trials <= 0:
        raise ValueError("--stage2-k-trials must be positive")
    if settings.stage2_probe_size <= 0:
        raise ValueError("--stage2-probe-size must be positive")
    if settings.cost_match_count < 0:
        raise ValueError("--cost-match-count must be non-negative")
    if settings.cost_match_max_attempts < 0:
        raise ValueError("--cost-match-max-attempts must be non-negative")
    if settings.cost_match_count > settings.cost_match_max_attempts:
        raise ValueError(
            "--cost-match-count cannot exceed --cost-match-max-attempts "
            f"(got {settings.cost_match_count} > {settings.cost_match_max_attempts})"
        )
    if settings.source == "json" and settings.config and not Path(settings.config).is_file():
        raise FileNotFoundError(f"final_eval JSON config does not exist: {settings.config}")
    if require_resume_for_search and settings.source == "search" and not settings.resume_from:
        raise ValueError("--source search requires --resume-from for standalone final_eval runs")
    if settings.resume_from and not Path(settings.resume_from).is_dir():
        raise FileNotFoundError(f"--resume-from directory does not exist: {settings.resume_from}")
    validate_supported_profile(settings.model_type, settings.dataset)
    if settings.stage2_rl_variant not in ("blb_v3", "legacy_v2"):
        raise ValueError("--stage2-rl-variant must be blb_v3 or legacy_v2")
    if settings.action_config and not Path(settings.action_config).is_file():
        raise FileNotFoundError(f"--action-config file does not exist: {settings.action_config}")
    if settings.random_enabled and settings.action_ranges:
        raise ValueError("--random/--enable-random cannot be combined with --action-range/--range")
    if settings.blb_rescale_invoker_kind not in ("heuristic", "in_process", "subprocess", "stub"):
        raise ValueError("--rescale-invoker-kind must be heuristic, in_process, subprocess, or stub")
    if settings.blb_rescale_optimizer_root and not Path(settings.blb_rescale_optimizer_root).is_dir():
        raise FileNotFoundError(f"--rescale-optimizer-root does not exist: {settings.blb_rescale_optimizer_root}")


def _sanitize_component(value: str, fallback: str = "run") -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = text.strip("._-")
    return text or fallback


def final_eval_output_run_dir(
    settings: FinalEvalSettings,
    *,
    source_run_dir: str = "",
    timestamp_if_needed: bool = True,
) -> Path:
    root = Path(settings.output_root)
    if not root.is_absolute():
        root = REPO_ROOT / root

    if settings.run_name:
        run_id = _sanitize_component(settings.run_name)
    elif source_run_dir:
        run_id = _sanitize_component(Path(source_run_dir).name, fallback="training_run")
    elif timestamp_if_needed:
        run_id = time.strftime("%Y%m%d_%H%M%S")
    else:
        run_id = "default"

    return root / settings.dataset / settings.algorithm / run_id


def final_eval_results_dir(
    settings: FinalEvalSettings,
    *,
    source_run_dir: str = "",
    timestamp_if_needed: bool = True,
) -> Path:
    return final_eval_output_run_dir(
        settings,
        source_run_dir=source_run_dir,
        timestamp_if_needed=timestamp_if_needed,
    ) / "final_eval"
