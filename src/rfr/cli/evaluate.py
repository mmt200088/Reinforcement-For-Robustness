"""Standalone launcher for one selected search configuration."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, List

from rfr.preparation.data.protocol import validate_supported_profile

from .evaluation_config import (
    REPO_ROOT,
    FinalEvalSettings,
    final_eval_output_run_dir,
    format_command,
    parse_final_eval_settings,
)


BASE_MODEL_BY_TYPE = {
    "bert-base": {
        "mrpc": "textattack/bert-base-uncased-MRPC",
        "rte": "textattack/bert-base-uncased-RTE",
        "sst2": "textattack/bert-base-uncased-SST-2",
    },
    "bert-large": {
        "mrpc": "yoshitomo-matsubara/bert-large-uncased-mrpc",
        "rte": "yoshitomo-matsubara/bert-large-uncased-rte",
        "sst2": "yoshitomo-matsubara/bert-large-uncased-sst2",
    },
}


def _base_model(model_type: str, dataset: str) -> str:
    validate_supported_profile(model_type, dataset)
    return BASE_MODEL_BY_TYPE[model_type][dataset]


def build_command(settings: FinalEvalSettings) -> List[str]:
    output_dir = final_eval_output_run_dir(settings)
    backend = "ppo" if settings.algorithm == "rl" else settings.algorithm
    return [
        sys.executable,
        "-m",
        "rfr.cli.run",
        "--base_model",
        _base_model(settings.model_type, settings.dataset),
        "--data_path",
        settings.dataset,
        "--output_dir",
        str(output_dir),
        "--search_best_config_path",
        settings.config,
        "--batch_size",
        str(settings.batch_size),
        "--final_eval_repeat_n",
        str(settings.repeat),
        "--final_eval_random_seed",
        str(settings.random_seed),
        "--skip_stage1_rl",
        "true",
        "--skip_noise_rl",
        "true",
        "--skip_final_eval",
        "false",
        "--final_eval_only",
        "true",
        "--stage1_rl_episodes",
        "51000",
        "--stage2_rl_episodes",
        "150000",
        "--stage1_rl_episodes_specified",
        "false",
        "--stage2_rl_episodes_specified",
        "false",
        "--ppo_update_interval",
        "120",
        "--blb_v3_search_backend",
        backend,
    ]


def log_path_for(settings: FinalEvalSettings, output_dir: Path) -> Path:
    name = Path(settings.logfile or "final_eval.log").name or "final_eval.log"
    return output_dir / "logs" / name


def configuration_lines(
        settings: FinalEvalSettings,
        output_dir: Path,
        command: List[str],
        ) -> List[str]:
    return [
        "selected-configuration final evaluation:",
        f"  config: {settings.config}",
        f"  algorithm: {settings.algorithm}",
        f"  model: {settings.model_type}",
        f"  dataset: {settings.dataset}",
        f"  repeat: {settings.repeat}",
        f"  batch_size: {settings.batch_size}",
        f"  output_dir: {output_dir}",
        f"  command: {format_command(command)}",
    ]


def launch_background(
        settings: FinalEvalSettings,
        command: List[str],
        output_dir: Path,
        *,
        popen_factory: Callable = subprocess.Popen,
        ) -> dict[str, Path | int]:
    log_path = log_path_for(settings, output_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write(f"Launcher time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        for line in configuration_lines(settings, output_dir, command):
            log_file.write(line + "\n")
        log_file.write("=" * 80 + "\n\n")
        log_file.flush()
        process = popen_factory(
            command,
            cwd=str(REPO_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid = int(process.pid)
    for path in (output_dir / "run.pid", output_dir / "final_eval.pid"):
        path.write_text(f"{pid}\n", encoding="utf-8")
    return {"pid": pid, "log_path": log_path, "output_dir": output_dir}


def main(argv: List[str] | None = None) -> int:
    try:
        settings = parse_final_eval_settings(
            list(argv if argv is not None else sys.argv[1:])
        )
    except (OSError, ValueError) as exc:
        print(f"final_eval: error: {exc}", file=sys.stderr)
        return 2
    output_dir = final_eval_output_run_dir(settings)
    command = build_command(settings)
    for line in configuration_lines(settings, output_dir, command):
        print(line)
    if settings.dry_run:
        return 0
    if settings.foreground:
        output_dir.mkdir(parents=True, exist_ok=True)
        return int(subprocess.run(command, cwd=str(REPO_ROOT), check=False).returncode)
    launched = launch_background(settings, command, output_dir)
    print(f"Started PID {launched['pid']}")
    print(f"Log: {launched['log_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
