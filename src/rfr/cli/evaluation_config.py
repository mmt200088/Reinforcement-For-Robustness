"""Command-line contract for selected-configuration evaluation."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import re
import time
from typing import Optional, Sequence

from rfr.common.runtime_error_reporter import format_command
from rfr.search.common.best_config import load_search_best_config


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs"


@dataclasses.dataclass(frozen=True)
class FinalEvalSettings:
    config: str
    algorithm: str
    model_type: str
    dataset: str
    batch_size: int = 64
    logfile: str = "final_eval.log"
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    run_name: str = ""
    repeat: int = 50
    random_seed: int = 42
    dry_run: bool = False
    foreground: bool = False


def resolve_repo_path(path_value: str) -> str:
    value = str(path_value or "").strip()
    if not value:
        return ""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_search.sh eval",
        description="Evaluate one completed search-best JSON on validation_full.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--logfile", default="final_eval.log")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_final_eval_settings(
        argv: Optional[Sequence[str]] = None,
        ) -> FinalEvalSettings:
    namespace = build_parser().parse_args(list(argv or ()))
    config_path = resolve_repo_path(namespace.config)
    payload = load_search_best_config(
        config_path,
        require_final_eval_eligible=True,
    )
    settings = FinalEvalSettings(
        config=config_path,
        algorithm=payload["algorithm"],
        model_type=payload["model_type"],
        dataset=payload["dataset"],
        batch_size=int(namespace.batch_size),
        logfile=str(namespace.logfile),
        output_root=resolve_repo_path(namespace.output_root),
        run_name=str(namespace.run_name),
        repeat=int(namespace.repeat),
        random_seed=int(namespace.random_seed),
        dry_run=bool(namespace.dry_run),
        foreground=bool(namespace.foreground),
    )
    validate_settings(settings)
    return settings


def validate_settings(settings: FinalEvalSettings) -> None:
    if settings.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if settings.repeat <= 0:
        raise ValueError("--repeat must be positive")
    if not Path(settings.config).is_file():
        raise FileNotFoundError(
            f"search-best JSON does not exist: {settings.config}"
        )


def _sanitize_component(value: str, fallback: str = "run") -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return text.strip("._-") or fallback


def final_eval_output_run_dir(
        settings: FinalEvalSettings,
        *,
        timestamp_if_needed: bool = True,
        ) -> Path:
    root = Path(settings.output_root)
    run_id = (
        _sanitize_component(settings.run_name)
        if settings.run_name
        else time.strftime("%Y%m%d_%H%M%S")
        if timestamp_if_needed
        else "default"
    )
    return (
        root
        / settings.algorithm
        / settings.model_type
        / settings.dataset
        / run_id
    )


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "REPO_ROOT",
    "FinalEvalSettings",
    "build_parser",
    "final_eval_output_run_dir",
    "format_command",
    "parse_final_eval_settings",
]
