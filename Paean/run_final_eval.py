from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

from .config import (
    DATASET_CHOICES,
    FinalEvalSettings,
    PRESET_DIR,
    REPO_ROOT,
    final_eval_output_run_dir,
    format_command,
    list_presets,
    parse_final_eval_settings,
)


def _base_model(model_type: str, dataset: str) -> str:
    if model_type == "bert-base":
        return {
            "wnli": "textattack/bert-base-uncased-WNLI",
            "rte": "textattack/bert-base-uncased-RTE",
            "cola": "textattack/bert-base-uncased-CoLA",
            "qnli": "textattack/bert-base-uncased-QNLI",
            "mrpc": "textattack/bert-base-uncased-MRPC",
            "sst2": "textattack/bert-base-uncased-SST-2",
            "stsb": "textattack/bert-base-uncased-STS-B",
        }[dataset]
    if model_type == "bert-large":
        models = {
            "mrpc": "yoshitomo-matsubara/bert-large-uncased-mrpc",
            "cola": "yoshitomo-matsubara/bert-large-uncased-cola",
            "stsb": "yoshitomo-matsubara/bert-large-uncased-stsb",
            "rte": "yoshitomo-matsubara/bert-large-uncased-rte",
            "sst2": "yoshitomo-matsubara/bert-large-uncased-sst2",
            "qnli": "yoshitomo-matsubara/bert-large-uncased-qnli",
        }
        if dataset not in models:
            raise ValueError("bert-large currently supports mrpc, cola, stsb, rte, sst2, qnli")
        return models[dataset]
    return {
        "cola": "PavanNeerudu/gpt2-finetuned-cola",
        "sst2": "PavanNeerudu/gpt2-finetuned-sst2",
        "mrpc": "PavanNeerudu/gpt2-finetuned-mrpc",
        "stsb": "PavanNeerudu/gpt2-finetuned-stsb",
        "qnli": "PavanNeerudu/gpt2-finetuned-qnli",
        "rte": "PavanNeerudu/gpt2-finetuned-rte",
        "wnli": "PavanNeerudu/gpt2-finetuned-wnli",
    }[dataset]


def build_command(settings: FinalEvalSettings) -> List[str]:
    output_dir = final_eval_output_run_dir(
        settings,
        source_run_dir=settings.resume_from,
        timestamp_if_needed=True,
    )
    base_model = _base_model(settings.model_type, settings.dataset)
    common = [
        "--base_model",
        base_model,
        "--data_path",
        settings.dataset,
        "--output_dir",
        str(output_dir),
        "--batch_size",
        str(settings.batch_size),
        "--micro_batch_size",
        str(settings.batch_size),
        "--num_epochs",
        "1",
        "--learning_rate",
        "2e-4",
        "--cutoff_len",
        "256",
        "--val_set_size",
        "120",
        "--eval_step",
        "80",
        "--adapter_name",
        "lora",
        "--target_modules",
        '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]',
        "--use_ist",
        "--final_eval_config_source",
        settings.source,
        "--final_eval_config_path",
        settings.config,
        "--manual_stage1_gelu",
        settings.manual_stage1_gelu,
        "--manual_stage1_softmax",
        settings.manual_stage1_softmax,
        "--manual_stage2_noise",
        settings.manual_stage2_noise,
        "--final_eval_random_seed",
        str(settings.random_seed),
        "--final_eval_permutation_trials",
        str(settings.perm_trials),
        "--final_eval_cost_equivalent_trials",
        str(settings.cost_trials),
        "--final_eval_budget_equivalent_trials",
        str(settings.budget_trials),
        "--final_eval_stage1_budget_trials",
        str(settings.stage1_budget_trials),
        "--final_eval_stage2_budget_trials",
        str(settings.stage2_budget_trials),
        "--final_eval_random_enabled",
        "true" if settings.random_enabled else "false",
        "--final_eval_repeat_n",
        str(settings.repeat),
        "--final_eval_action_config",
        settings.action_config,
        "--final_eval_action_ranges",
        json.dumps(list(settings.action_ranges)),
        "--final_eval_action_fixed",
        json.dumps(list(settings.action_fixed)),
        "--final_eval_cost_match_count",
        str(settings.cost_match_count),
        "--final_eval_cost_match_max_attempts",
        str(settings.cost_match_max_attempts),
        "--final_eval_glue_submission_enabled",
        "true" if settings.glue_submission_enabled else "false",
        "--final_eval_glue_submission_seed",
        str(settings.glue_submission_seed),
        "--blb_v3_rescale_invoker_kind",
        settings.blb_rescale_invoker_kind,
        "--blb_v3_inproc_rescale_optimizer_root",
        settings.blb_rescale_optimizer_root,
        "--final_eval_require_rescale_optimizer",
        "true" if settings.require_rescale_optimizer else "false",
        "--skip_noise_rl",
        "true",
        "--skip_stage1_rl",
        "true",
        "--skip_final_eval",
        "false",
        "--final_eval_only",
        "true",
        "--resume_run_dir",
        settings.resume_from,
        "--stage1_accuracy_tolerance",
        str(settings.stage1_accuracy_tolerance),
        "--stage2_limit_tolerance",
        str(settings.stage2_limit_tolerance),
        "--stage2_stability_tolerance",
        str(settings.stage2_stability_tolerance),
        "--stage2_k_trials",
        str(settings.stage2_k_trials),
        "--stage2_probe_size",
        str(settings.stage2_probe_size),
    ]

    if settings.algorithm == "rl":
        return [
            sys.executable,
            str(REPO_ROOT / "rl_tune.py"),
            *common,
            "--stage1_rl_episodes",
            "51000",
            "--stage2_rl_episodes",
            "40000",
            "--stage1_rl_episodes_specified",
            "false",
            "--stage2_rl_episodes_specified",
            "false",
            "--ppo_update_interval",
            "120",
            "--stage2_rl_variant",
            settings.stage2_rl_variant,
        ]

    return [
        sys.executable,
        str(REPO_ROOT / "rl_tune_genetic.py"),
        *common,
        "--search_backend",
        settings.algorithm,
    ]


def log_path_for(settings: FinalEvalSettings, output_dir: Path) -> Path:
    logfile_name = Path(str(settings.logfile or "final_eval.log")).name
    if not logfile_name:
        logfile_name = "final_eval.log"
    return output_dir / "logs" / logfile_name


def configuration_lines(
    settings: FinalEvalSettings,
    output_dir: Path,
    command: List[str],
    *,
    include_command: bool = True,
) -> List[str]:
    lines = [
        "final_eval standalone configuration:",
        f"  dataset: {settings.dataset}",
        f"  algorithm: {settings.algorithm}",
        f"  source: {settings.source}",
        f"  config: {settings.config}",
        f"  random_enabled: {settings.random_enabled}",
    ]
    if settings.action_config:
        lines.append(f"  action_config: {settings.action_config}")
    if settings.action_ranges:
        lines.append(f"  action_ranges: {list(settings.action_ranges)}")
    if settings.action_fixed:
        lines.append(f"  action_fixed: {list(settings.action_fixed)}")
    lines.extend(
        [
            f"  blb_rescale_invoker_kind: {settings.blb_rescale_invoker_kind}",
        ]
    )
    if settings.blb_rescale_optimizer_root:
        lines.append(f"  blb_rescale_optimizer_root: {settings.blb_rescale_optimizer_root}")
    lines.append(f"  require_rescale_optimizer: {settings.require_rescale_optimizer}")
    if settings.resume_from:
        lines.append(f"  resume_from: {settings.resume_from}")
    lines.append(f"  output_dir: {output_dir}")
    if include_command:
        lines.append(f"  command: {format_command(command)}")
    return lines


def print_configuration(
    settings: FinalEvalSettings,
    output_dir: Path,
    command: List[str],
    *,
    include_command: bool = True,
) -> None:
    for line in configuration_lines(
        settings,
        output_dir,
        command,
        include_command=include_command,
    ):
        print(line)


def launch_background(
    settings: FinalEvalSettings,
    command: List[str],
    output_dir: Path,
    *,
    popen_factory: Callable = subprocess.Popen,
) -> Dict[str, Path | int]:
    log_path = log_path_for(settings, output_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_base_dir = output_dir.parent
    latest_base_dir.mkdir(parents=True, exist_ok=True)

    command_text = format_command(command)
    (log_path.parent / "launch_command.txt").write_text(command_text + "\n", encoding="utf-8")

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write(f"Launcher time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        for line in configuration_lines(
            settings,
            output_dir,
            command,
            include_command=True,
        ):
            log_file.write(line + "\n")
        log_file.write("=" * 80 + "\n\n")
        log_file.flush()
        proc = popen_factory(
            command,
            cwd=str(REPO_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    pid = int(proc.pid)
    for pid_file in (output_dir / "run.pid", output_dir / "final_eval.pid"):
        pid_file.write_text(f"{pid}\n", encoding="utf-8")
    (latest_base_dir / "LATEST_RUN_DIR").write_text(str(output_dir) + "\n", encoding="utf-8")
    (latest_base_dir / "LATEST_PID").write_text(f"{pid}\n", encoding="utf-8")
    return {
        "pid": pid,
        "log_path": log_path,
        "output_dir": output_dir,
        "latest_base_dir": latest_base_dir,
    }


def main(argv: List[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if "--list-presets" in argv:
        print("Available final_eval presets:")
        for name in list_presets(PRESET_DIR):
            path = PRESET_DIR / f"{name}.conf"
            first = ""
            try:
                first = path.read_text(encoding="utf-8-sig").splitlines()[0].lstrip("# ")
            except Exception:
                pass
            print(f"  {name:<30} {first}")
        return 0

    try:
        settings = parse_final_eval_settings(argv)
        command = build_command(settings)
    except ValueError as exc:
        print(f"final_eval: error: {exc}", file=sys.stderr)
        return 2
    output_dir = final_eval_output_run_dir(
        settings,
        source_run_dir=settings.resume_from,
        timestamp_if_needed=True,
    )

    if settings.dry_run:
        print_configuration(settings, output_dir, command, include_command=True)
        return 0

    if not settings.foreground:
        launch = launch_background(settings, command, output_dir)
        print("已在后台启动 final_eval。")
        print(f"  进程号（PID）：{launch['pid']}")
        print(f"  输出目录：{launch['output_dir']}")
        print(f"  查看日志：tail -f {launch['log_path']}")
        print(f"  LATEST_RUN_DIR：{launch['latest_base_dir'] / 'LATEST_RUN_DIR'}")
        print(f"  LATEST_PID：{launch['latest_base_dir'] / 'LATEST_PID'}")
        print(f"  优雅停止（Graceful Stop）：kill -INT {launch['pid']}")
        return 0

    print_configuration(settings, output_dir, command, include_command=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
