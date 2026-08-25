from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Dict, List

from rfr.preparation.data.protocol import validate_supported_profile
from .evaluation_config import (
    PRESET_DIR,
    REPO_ROOT,
    FinalEvalSettings,
    final_eval_output_run_dir,
    format_command,
    list_presets,
    parse_final_eval_settings,
)

_ACTION_BATCH_SCHEMA = "paean_action_batch_v1"

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
    model_family = str(model_type).strip().lower()
    task = str(dataset).strip().lower()
    validate_supported_profile(model_family, task)
    return BASE_MODEL_BY_TYPE[model_family][task]


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
        json.dumps(settings.action_ranges),
        "--final_eval_action_fixed",
        json.dumps(settings.action_fixed),
        "--final_eval_cost_match_count",
        str(settings.cost_match_count),
        "--final_eval_cost_match_max_attempts",
        str(settings.cost_match_max_attempts),
        "--blb_v3_inproc_rescale_optimizer_root",
        settings.blb_rescale_optimizer_root,
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

    backend = "ppo" if settings.algorithm == "rl" else settings.algorithm
    return [
        sys.executable,
        "-m",
        "rfr.cli.run",
        *common,
        "--stage1_rl_episodes",
        "51000",
        "--stage2_rl_episodes",
        "0",
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
    logfile_name = Path(str(settings.logfile or "final_eval.log")).name
    if not logfile_name:
        logfile_name = "final_eval.log"
    return output_dir / "logs" / logfile_name


def _action_range_value_count(range_spec: str) -> int:
    if "=" not in range_spec:
        return 1
    _name, raw_values = range_spec.split("=", 1)
    values = [value.strip() for value in raw_values.split(",") if value.strip()]
    return max(1, len(values))


def _action_config_candidate_count(action_config: str) -> int:
    if not action_config:
        return 1

    path = Path(action_config).expanduser()
    if not path.is_file():
        return 1

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 1

    if not isinstance(payload, dict):
        return 1
    if payload.get("schema_version") != _ACTION_BATCH_SCHEMA:
        return 1

    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return 1
    return max(1, len(candidates))


def estimate_workload(settings: FinalEvalSettings) -> Dict[str, object]:
    selected_config_count = _action_config_candidate_count(settings.action_config)
    for range_spec in settings.action_ranges:
        selected_config_count *= _action_range_value_count(str(range_spec))

    legacy_random_control_count = 0
    if settings.random_enabled:
        legacy_random_control_count = sum(
            max(0, int(count))
            for count in (
                settings.perm_trials,
                settings.cost_trials,
                settings.budget_trials,
                settings.stage1_budget_trials,
                settings.stage2_budget_trials,
            )
        )

    cost_matched_random_count = max(0, int(settings.cost_match_count))
    total_config_count = (
        selected_config_count
        + legacy_random_control_count
        + cost_matched_random_count
    )
    repeat = max(1, int(settings.repeat))
    total_repeated_evaluations = total_config_count * repeat
    gpu_parallelism_candidate = total_config_count > 1

    return {
        "action_range_dimensions": len(settings.action_ranges),
        "selected_config_count": selected_config_count,
        "legacy_random_control_count": legacy_random_control_count,
        "cost_matched_random_count": cost_matched_random_count,
        "total_config_count": total_config_count,
        "repeat": repeat,
        "total_repeated_evaluations": total_repeated_evaluations,
        "batch_size": int(settings.batch_size),
        "stage2_k_trials": int(settings.stage2_k_trials),
        "stage2_probe_size": int(settings.stage2_probe_size),
        "launcher_processes": 1,
        "gpu_parallelism_candidate": gpu_parallelism_candidate,
        "gpu_parallelism_hint": (
            "independent final-eval configs can be scheduled across GPUs"
            if gpu_parallelism_candidate
            else "single final-eval config"
        ),
    }


def configuration_lines(
    settings: FinalEvalSettings,
    output_dir: Path,
    command: List[str],
    *,
    include_command: bool = True,
) -> List[str]:
    workload = estimate_workload(settings)
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
    if settings.blb_rescale_optimizer_root:
        lines.append(f"  blb_rescale_optimizer_root: {settings.blb_rescale_optimizer_root}")
    if settings.resume_from:
        lines.append(f"  resume_from: {settings.resume_from}")
    lines.append(f"  output_dir: {output_dir}")
    lines.extend(
        [
            "  workload:",
            f"    selected_configs: {workload['selected_config_count']}",
            f"    action_range_dimensions: {workload['action_range_dimensions']}",
            f"    legacy_random_controls: {workload['legacy_random_control_count']}",
            f"    cost_matched_random_configs: {workload['cost_matched_random_count']}",
            f"    total_configs: {workload['total_config_count']}",
            f"    repeat: {workload['repeat']}",
            f"    total_repeated_evaluations: {workload['total_repeated_evaluations']}",
            f"    batch_size: {workload['batch_size']}",
            f"    stage2_k_trials: {workload['stage2_k_trials']}",
            f"    stage2_probe_size: {workload['stage2_probe_size']}",
            f"    launcher_processes: {workload['launcher_processes']}",
            "    gpu_parallelism_candidate: "
            f"{str(workload['gpu_parallelism_candidate']).lower()}",
            f"    gpu_parallelism_hint: {workload['gpu_parallelism_hint']}",
        ]
    )
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


def _read_preset_summary(path: Path) -> str:
    try:
        with path.open(encoding="utf-8-sig") as handle:
            return handle.readline().rstrip("\r\n").lstrip("# ")
    except Exception:
        return ""


def main(argv: List[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if "--list-presets" in argv:
        print("Available final_eval presets:")
        for name in list_presets(PRESET_DIR):
            path = PRESET_DIR / f"{name}.conf"
            first = _read_preset_summary(path)
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
