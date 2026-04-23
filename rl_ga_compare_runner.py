import argparse
import ctypes
import gc
import json
import os
import random
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from runtime_error_reporter import (
    clear_error_summary,
    read_text_tail,
    write_error_summary,
)


TARGET_MODULES_LITERAL = "[\"q_proj\", \"k_proj\", \"v_proj\", \"up_proj\", \"down_proj\"]"
STAGE1_RL_CHECKPOINT_FILENAME = "stage1_rl_checkpoint.pt"
NOISE_STAGE_CHECKPOINT_FILENAME = "noise_rl_checkpoint.pt"
GA_STAGE1_CHECKPOINT_FILENAME = "ga_stage1_checkpoint.pt"
GA_STAGE2_CHECKPOINT_FILENAME = "ga_stage2_checkpoint.pt"
DEFAULT_POLL_SECONDS = 15
LINUX_PR_SET_PDEATHSIG = 1

DATASET_METRIC_SHORT_NAMES = {
    "sst2": ["Acc."],
    "qnli": ["Acc."],
    "mnli": ["M-Acc.", "MM-Acc."],
    "cola": ["MCC"],
    "stsb": ["Pear.", "Spear."],
    "mrpc": ["Acc.", "F1"],
    "rte": ["Acc."],
    "wnli": ["Acc."],
}


@dataclass
class ChildRunSpec:
    algorithm: str
    entrypoint: str
    run_dir: Path
    log_path: Path
    command: List[str]
    env_overrides: Dict[str, str]
    process: Optional[subprocess.Popen] = None


@dataclass
class CompareSideConfig:
    skip_stage1_search: bool = False
    skip_noise_search: bool = False
    final_eval_config_source: str = "search"
    final_eval_config_path: str = ""


@dataclass
class EvaluationOnlySideSpec:
    algorithm: str
    run_dir: Path
    side_config: CompareSideConfig
    stage1_input_kind: str
    stage2_input_kind: str
    stage1_input_path: Optional[Path] = None
    stage2_input_path: Optional[Path] = None
    source_metadata: Optional[dict] = None


class CompareRunnerError(RuntimeError):
    pass


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now_ts()}] {msg}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> int:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    return seed


def cleanup_cuda_memory() -> None:
    """Release Python references and cached CUDA blocks between compare sides."""
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def release_compare_evaluator(evaluator) -> None:
    if evaluator is None:
        cleanup_cuda_memory()
        return
    model = getattr(evaluator, "model", None)
    if model is not None:
        try:
            model.to("cpu")
        except Exception:
            pass
    cleanup_cuda_memory()


def _build_parent_death_preexec_fn():
    if os.name != "posix":
        return None

    parent_pid = os.getpid()

    def _preexec():
        try:
            libc = ctypes.CDLL(None)
            result = libc.prctl(
                LINUX_PR_SET_PDEATHSIG,
                signal.SIGKILL,
                0,
                0,
                0,
            )
            if result != 0:
                return
            if os.getppid() != parent_pid:
                os.kill(os.getpid(), signal.SIGKILL)
        except Exception:
            # Best effort only. On non-Linux POSIX systems, or when prctl is
            # unavailable, we simply fall back to the existing behavior.
            return

    return _preexec


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_cuda_value(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def split_cuda_visible_devices(raw: Optional[str]) -> Tuple[Optional[str], Optional[str], List[str]]:
    warnings: List[str] = []
    raw = normalize_cuda_value(raw)
    if not raw:
        warnings.append("未检测到 CUDA_VISIBLE_DEVICES；RL 与 GA 将继承当前环境，可能共享同一设备。")
        return None, None, warnings

    devices = [item.strip() for item in raw.split(",") if item.strip()]
    if len(devices) < 2:
        warnings.append(
            f"CUDA_VISIBLE_DEVICES={raw} 仅包含 {len(devices)} 个可见设备；RL 与 GA 将共享该设置。"
        )
        return raw, raw, warnings

    rl_devices = devices[0]
    ga_devices = devices[1]
    if len(devices) > 2:
        warnings.append(
            f"检测到 {len(devices)} 个可见设备；对比模式默认只分配第一个给 RL（{rl_devices}），"
            f"第二个给 GA（{ga_devices}），其余设备不自动使用。"
        )
    return rl_devices, ga_devices, warnings


def normalize_compare_config_source(raw_value: str, *, flag_name: str) -> str:
    source = str(raw_value or "search").strip().lower()
    if source not in ("search", "json"):
        raise CompareRunnerError(
            f"{flag_name} must be one of: search, json."
        )
    return source


def normalize_compare_side_config(
    *,
    label: str,
    skip_stage1_search: bool,
    skip_noise_search: bool,
    final_eval_config_source: str,
    final_eval_config_path: str,
) -> CompareSideConfig:
    source = normalize_compare_config_source(
        final_eval_config_source,
        flag_name=f"{label.lower()} final-eval source",
    )
    path = str(final_eval_config_path or "").strip()

    need_json = bool(skip_stage1_search) or bool(skip_noise_search)
    if need_json:
        if source == "search":
            raise CompareRunnerError(
                f"{label} 跳过 Stage-1 或 Stage-2 搜索时，final-eval 配置来源不能是 search；"
                "请改用 json。"
            )
    else:
        if source != "search":
            raise CompareRunnerError(
                f"{label} 未跳过任何搜索阶段时，final-eval 配置来源必须为 search。"
            )
        if path:
            raise CompareRunnerError(
                f"{label} 未跳过任何搜索阶段时，不应提供 final-eval JSON 配置路径。"
            )

    if source == "json" and not path:
        raise CompareRunnerError(
            f"{label} 使用 JSON 配置时，必须提供配置文件路径。"
        )

    return CompareSideConfig(
        skip_stage1_search=bool(skip_stage1_search),
        skip_noise_search=bool(skip_noise_search),
        final_eval_config_source=source,
        final_eval_config_path=path,
    )


def final_eval_json(run_dir: Path, dataset: str) -> Path:
    return run_dir / "final_eval" / f"final_eval_results_{dataset}.json"


def final_eval_json_matches_protocol(
    obj: Optional[dict],
    repeat_n: int,
    *,
    expect_random_groups: bool = True,
) -> bool:
    if not obj:
        return False
    repeat_n = max(1, int(repeat_n))
    protocol = obj.get("evaluation_protocol") or {}
    if int(protocol.get("version", 0) or 0) < 2:
        return False
    if protocol.get("baseline") != "single_clean_validation_full":
        return False
    if int(protocol.get("noisy_repeat_n", 0) or 0) != repeat_n:
        return False
    random_groups_mode = protocol.get("random_groups")
    if expect_random_groups:
        if random_groups_mode != "enabled":
            return False
    else:
        if random_groups_mode != "disabled":
            return False
    if obj.get("baseline_repeat_evaluation") is not None:
        return False

    baseline = obj.get("baseline") or {}
    if baseline.get("evaluation_n") is not None:
        return False

    if repeat_n > 1:
        optimized = obj.get("optimized") or {}
        if int(optimized.get("evaluation_n", 0) or 0) != repeat_n:
            return False
        if expect_random_groups:
            for result in obj.get("random_results") or []:
                if int(result.get("evaluation_n", 0) or 0) != repeat_n:
                    return False
    if not expect_random_groups and (obj.get("random_results") or []):
        return False
    return True


def stage1_ga_result_json(run_dir: Path) -> Path:
    return run_dir / "stage1" / "ga_search_results.json"


def stage2_ga_result_json(run_dir: Path) -> Path:
    return run_dir / "stage2_noise" / "noise_ga_search_results.json"


def stage1_rl_checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "stage1" / STAGE1_RL_CHECKPOINT_FILENAME


def stage2_rl_checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "stage2_noise" / "progress" / NOISE_STAGE_CHECKPOINT_FILENAME


def stage1_ga_checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "stage1" / GA_STAGE1_CHECKPOINT_FILENAME


def stage2_ga_checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "stage2_noise" / "progress" / GA_STAGE2_CHECKPOINT_FILENAME


def load_torch_checkpoint(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def clone_stage1_config(cfg: Optional[dict]) -> Optional[dict]:
    if cfg is None:
        return None
    out = {}
    if "gelu" in cfg:
        out["gelu"] = np.asarray(cfg["gelu"], dtype=int).tolist()
    if "softmax" in cfg:
        out["softmax"] = np.asarray(cfg["softmax"], dtype=int).tolist()
    for key in ("loss", "metric1", "metric2", "cost", "reward", "proxy_reward", "score", "penalty", "feasible"):
        if key in cfg:
            out[key] = to_jsonable(cfg[key])
    if "gelu" not in out or "softmax" not in out:
        return None
    return out


def clone_noise_config(cfg: Optional[dict]) -> Optional[dict]:
    if cfg is None:
        return None
    out = {}
    for key, value in cfg.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            out[key] = np.asarray(value, dtype=int).tolist()
        else:
            out[key] = to_jsonable(value)
    return out


def recover_stage1_search_best(run_dir: Path, algorithm: str) -> Tuple[Optional[dict], str]:
    if algorithm == "rl":
        ckpt = load_torch_checkpoint(stage1_rl_checkpoint_path(run_dir))
        if ckpt is not None:
            cfg = (
                ckpt.get("global_best_config")
                or ckpt.get("search_best_config")
                or ckpt.get("best_config")
            )
            cfg = clone_stage1_config(cfg)
            if cfg is not None:
                return cfg, "stage1_rl_checkpoint"
        return None, "stage1_rl_checkpoint_missing"

    result_obj = read_json(stage1_ga_result_json(run_dir))
    if result_obj is not None:
        cfg = clone_stage1_config(result_obj.get("best_config"))
        if cfg is not None:
            return cfg, "stage1_ga_search_results"

    ckpt = load_torch_checkpoint(stage1_ga_checkpoint_path(run_dir))
    if ckpt is not None:
        cfg = clone_stage1_config(ckpt.get("best_candidate"))
        if cfg is not None:
            return cfg, "stage1_ga_checkpoint"

    return None, "stage1_ga_artifact_missing"


def recover_stage2_search_best(run_dir: Path, algorithm: str) -> Tuple[Optional[dict], str]:
    if algorithm == "rl":
        ckpt = load_torch_checkpoint(stage2_rl_checkpoint_path(run_dir))
        if ckpt is not None:
            cfg = ckpt.get("incumbent_best_noise_config") or ckpt.get("best_noise_config")
            cfg = clone_noise_config(cfg)
            if cfg is not None:
                return cfg, "stage2_rl_checkpoint"
        return None, "stage2_rl_checkpoint_missing"

    result_obj = read_json(stage2_ga_result_json(run_dir))
    if result_obj is not None:
        cfg = (
            result_obj.get("best_noise_config")
            or result_obj.get("stable_search_best_noise_config")
            or (
                result_obj.get("selection_diagnostics", {}) or {}
            ).get("final_incumbent")
        )
        cfg = clone_noise_config(cfg)
        if cfg is not None:
            return cfg, "stage2_ga_search_results"

    ckpt = load_torch_checkpoint(stage2_ga_checkpoint_path(run_dir))
    if ckpt is not None:
        cfg = clone_noise_config(ckpt.get("incumbent"))
        if cfg is not None:
            return cfg, "stage2_ga_checkpoint"

    return None, "stage2_ga_artifact_missing"


def resolve_stage1_selected_config_from_artifacts(
    run_dir: Path, algorithm: str, dataset: str
) -> Tuple[Optional[dict], str]:
    obj = read_json(final_eval_json(run_dir, dataset))
    if obj is not None and obj.get("selected") is not None:
        return {
            "gelu": obj["selected"]["gelu"],
            "softmax": obj["selected"]["softmax"],
        }, "final_eval_selected"
    if obj is not None:
        optimized_stage1 = obj.get("optimized_stage1") or {}
        if optimized_stage1.get("gelu") is not None and optimized_stage1.get("softmax") is not None:
            return {
                "gelu": optimized_stage1["gelu"],
                "softmax": optimized_stage1["softmax"],
            }, "final_eval_optimized_stage1"
        optimized = obj.get("optimized") or {}
        if optimized.get("gelu") is not None and optimized.get("softmax") is not None:
            return {
                "gelu": optimized["gelu"],
                "softmax": optimized["softmax"],
            }, "final_eval_optimized"
    cfg, source = recover_stage1_search_best(run_dir, algorithm)
    return cfg, source


def default_final_eval_json_for_algorithm(algorithm: str) -> str:
    return (
        "glue_final_configs_best_genetic.json"
        if algorithm == "ga"
        else "glue_final_configs_best_ppo.json"
    )


def normalize_model_type(raw_value: str) -> str:
    value = str(raw_value or "").strip().lower().replace("_", "-")
    alias_map = {
        "bertbase": "bert-base",
        "bert-base": "bert-base",
        "bertlarge": "bert-large",
        "bert-large": "bert-large",
        "gpt2": "gpt-2",
        "gpt-2": "gpt-2",
    }
    normalized = alias_map.get(value)
    if normalized is None:
        raise CompareRunnerError(
            f"Unsupported model_type={raw_value!r}. Expected one of: bert-base, bert-large, gpt-2."
        )
    return normalized


def expected_total_layers(model_type: str) -> int:
    if model_type == "bert-large":
        return 24
    return 12


def compare_constraint_slug(
    stage1_accuracy_tolerance: float,
    stage2_limit_tolerance: float,
    stage2_stability_tolerance: float,
) -> str:
    return (
        f"s1t{stage1_accuracy_tolerance}"
        f"_s2t{stage2_limit_tolerance}"
        f"_s2st{stage2_stability_tolerance}"
    )


def persistent_run_dir_for_compare(
    *,
    persistent_root: Path,
    algorithm: str,
    model_type: str,
    dataset: str,
    stage1_accuracy_tolerance: float,
    stage2_limit_tolerance: float,
    stage2_stability_tolerance: float,
) -> Path:
    return (
        persistent_root
        / algorithm
        / model_type
        / dataset
        / compare_constraint_slug(
            stage1_accuracy_tolerance,
            stage2_limit_tolerance,
            stage2_stability_tolerance,
        )
    )


def infer_algorithm_family_from_path(path: Path) -> str:
    lower_path = str(path).lower().replace("\\", "/")
    parts = [part for part in lower_path.split("/") if part]
    for part in reversed(parts):
        if part in ("rl", "ga"):
            return part
    name = path.name.lower()
    if "genetic" in name or name.startswith("ga_") or "_ga." in name or "-ga." in name:
        return "ga"
    if "ppo" in name:
        return "rl"
    return "unknown"


def _resolve_model_section(config_map: dict, model_type: str, *, config_path: Path) -> dict:
    if model_type in config_map and isinstance(config_map[model_type], dict):
        return config_map[model_type]
    if any(key in config_map for key in ("bert-base", "bert-large", "gpt-2")):
        raise CompareRunnerError(
            f"Model variant '{model_type}' not found in '{config_path}'."
        )
    if model_type != "bert-base":
        raise CompareRunnerError(
            f"Config file '{config_path}' uses the legacy flat schema and only supports bert-base."
        )
    return config_map


def _stage1_gelu_softmax_from_dataset_entry(dataset_obj: object) -> Tuple[Optional[object], Optional[object]]:
    """Resolve gelu/softmax from a per-dataset entry.

    Supports:
    - Legacy template: ``{ "gelu": [...], "softmax": [...] }`` at dataset root.
    - Unified glue file: ``{ "stage1": { "gelu": ..., "softmax": ... }, "stage2": ... }``.
    """
    if not isinstance(dataset_obj, dict):
        return None, None
    if "gelu" in dataset_obj and "softmax" in dataset_obj:
        return dataset_obj.get("gelu"), dataset_obj.get("softmax")
    stage1 = dataset_obj.get("stage1")
    if isinstance(stage1, dict) and "gelu" in stage1 and "softmax" in stage1:
        return stage1.get("gelu"), stage1.get("softmax")
    return None, None


def _stage2_noise_block_from_dataset_entry(dataset_obj: object) -> Optional[dict]:
    """Resolve the noise scaling dict from a per-dataset entry.

    Supports legacy flat keys (``x``, ``wq``, ...) at dataset root, or unified glue
    ``stage2`` nested object with the same short keys.
    """
    required_keys = (
        "x",
        "wq",
        "wk",
        "wv",
        "wo",
        "wffn1",
        "wffn2",
    )
    if not isinstance(dataset_obj, dict):
        return None
    if all(key in dataset_obj for key in required_keys):
        return dataset_obj
    stage2 = dataset_obj.get("stage2")
    if isinstance(stage2, dict) and all(key in stage2 for key in required_keys):
        return stage2
    return None


def validate_stage1_config_template(path: Path, *, dataset: str, model_type: str) -> None:
    obj = read_json(path)
    if not isinstance(obj, dict):
        raise CompareRunnerError(f"Invalid Stage-1 JSON file: {path}")
    obj = dict(obj)
    obj.pop("_comment", None)
    section = _resolve_model_section(obj, model_type, config_path=path)
    if dataset not in section:
        raise CompareRunnerError(
            f"Dataset '{dataset}' not found under '{model_type}' in Stage-1 JSON '{path}'."
        )
    dataset_obj = section[dataset]
    gelu, softmax = _stage1_gelu_softmax_from_dataset_entry(dataset_obj)
    if gelu is None or softmax is None:
        raise CompareRunnerError(
            f"Stage-1 JSON '{path}' is missing gelu/softmax for dataset '{dataset}'."
        )


def validate_stage2_config_template(path: Path, *, dataset: str, model_type: str) -> None:
    obj = read_json(path)
    if not isinstance(obj, dict):
        raise CompareRunnerError(f"Invalid Stage-2 JSON file: {path}")
    obj = dict(obj)
    obj.pop("_comment", None)
    section = _resolve_model_section(obj, model_type, config_path=path)
    if dataset not in section:
        raise CompareRunnerError(
            f"Dataset '{dataset}' not found under '{model_type}' in Stage-2 JSON '{path}'."
        )
    dataset_obj = section[dataset]
    noise_block = _stage2_noise_block_from_dataset_entry(dataset_obj)
    if noise_block is None:
        raise CompareRunnerError(
            f"Stage-2 JSON '{path}' has an invalid dataset payload for '{dataset}' "
            f"(expected flat x/wq/... or unified stage2.x / stage2.wq / ...)."
        )
    required_keys = (
        "x",
        "wq",
        "wk",
        "wv",
        "wo",
        "wffn1",
        "wffn2",
    )
    missing = [key for key in required_keys if key not in noise_block]
    if missing:
        raise CompareRunnerError(
            f"Stage-2 JSON '{path}' is missing keys for dataset '{dataset}': {missing}."
        )


def _extract_stage1_result_layer_count(obj: dict) -> Optional[int]:
    for candidate in (
        (obj.get("selected") or {}).get("gelu"),
        (obj.get("optimized") or {}).get("gelu"),
        (obj.get("optimized_stage1") or {}).get("gelu"),
        (obj.get("baseline") or {}).get("gelu"),
        (obj.get("no_approx") or {}).get("gelu"),
    ):
        if isinstance(candidate, list) and candidate:
            return len(candidate)
    return None


def _extract_stage2_result_layer_count(obj: dict) -> Optional[int]:
    for stage1_candidate in (
        obj.get("fixed_stage1_config") or {},
        obj.get("optimized_stage1") or {},
    ):
        gelu = stage1_candidate.get("gelu")
        if isinstance(gelu, list) and gelu:
            return len(gelu)

    for result_key in ("selected", "optimized"):
        noise_cfg = (obj.get(result_key) or {}).get("noise_config") or {}
        for value in noise_cfg.values():
            if isinstance(value, list) and value:
                return len(value)
    optimized_stage2 = obj.get("optimized_stage2") or {}
    for value in optimized_stage2.values():
        if isinstance(value, list) and value:
            return len(value)
    return None


def validate_stage_result_json(
    path: Path,
    *,
    stage_label: str,
    dataset: str,
    model_type: str,
) -> dict:
    obj = read_json(path)
    if not isinstance(obj, dict):
        raise CompareRunnerError(f"Invalid {stage_label} result JSON: {path}")
    actual_dataset = obj.get("dataset")
    if actual_dataset and actual_dataset != dataset:
        raise CompareRunnerError(
            f"{stage_label} result JSON '{path}' belongs to dataset '{actual_dataset}', expected '{dataset}'."
        )

    expected_layers = expected_total_layers(model_type)
    actual_layers = (
        _extract_stage1_result_layer_count(obj)
        if stage_label == "stage1"
        else _extract_stage2_result_layer_count(obj)
    )
    if actual_layers is not None and actual_layers != expected_layers:
        raise CompareRunnerError(
            f"{stage_label} result JSON '{path}' has layer count {actual_layers}, "
            f"which does not match model_type='{model_type}' ({expected_layers} layers)."
        )
    return obj


def materialize_result_json(
    *,
    stage_label: str,
    src_path: Path,
    run_dir: Path,
    dataset: str,
) -> Path:
    obj = read_json(src_path)
    if not isinstance(obj, dict):
        raise CompareRunnerError(f"Invalid {stage_label} result JSON: {src_path}")
    dest_path = (
        final_eval_json(run_dir, dataset)
        if stage_label == "stage1"
        else final_eval_json(run_dir, dataset)
    )
    write_json(dest_path, obj)
    return dest_path


def validate_persistent_run_dir(
    *,
    run_dir: Path,
    algorithm: str,
    model_type: str,
    dataset: str,
) -> dict:
    metadata_path = run_dir / "metadata.json"
    metadata = read_json(metadata_path)
    if metadata is None:
        raise CompareRunnerError(
            f"Persistent run dir '{run_dir}' is missing metadata.json."
        )
    actual_algorithm = str(metadata.get("algorithm") or "").strip().lower()
    actual_model_type = str(metadata.get("model_type") or "").strip().lower()
    actual_dataset = str(metadata.get("dataset") or "").strip().lower()
    if actual_algorithm != algorithm:
        raise CompareRunnerError(
            f"Persistent run dir '{run_dir}' belongs to algorithm '{actual_algorithm}', expected '{algorithm}'."
        )
    if actual_model_type != model_type:
        raise CompareRunnerError(
            f"Persistent run dir '{run_dir}' belongs to model_type '{actual_model_type}', expected '{model_type}'."
        )
    if actual_dataset != dataset:
        raise CompareRunnerError(
            f"Persistent run dir '{run_dir}' belongs to dataset '{actual_dataset}', expected '{dataset}'."
        )
    return metadata


def ensure_persistent_side_has_compare_artifacts(
    *,
    run_dir: Path,
    algorithm: str,
    dataset: str,
) -> None:
    if not final_eval_json(run_dir, dataset).is_file():
        stage1_cfg, _ = recover_stage1_search_best(run_dir, algorithm)
        if stage1_cfg is None:
            raise CompareRunnerError(
                f"Persistent run dir '{run_dir}' has no usable Stage-1 compare artifact."
            )
    if not final_eval_json(run_dir, dataset).is_file():
        stage2_cfg, _ = recover_stage2_search_best(run_dir, algorithm)
        if stage2_cfg is None:
            raise CompareRunnerError(
                f"Persistent run dir '{run_dir}' has no usable Stage-2 compare artifact."
            )


def _extract_repeat_evaluation(obj: dict) -> Tuple[Optional[dict], List[dict]]:
    repeat_obj = obj.get("optimized_repeat_evaluation") or obj.get("repeat_evaluation") or {}
    stats = repeat_obj.get("stats")
    trials = repeat_obj.get("trials") or []
    if stats is None:
        return None, []
    return stats, trials


def _variance_from_stats(stats: Optional[dict], key: str) -> Optional[float]:
    if not stats:
        return None
    std_value = stats.get(f"{key}_std")
    if std_value is None:
        return None
    std_value = float(std_value)
    return float(std_value * std_value)


def _build_stage2_repeat_summary(metric_names: List[str], rl_side: dict, ga_side: dict) -> Optional[dict]:
    rl_stats = rl_side.get("repeat_stats")
    ga_stats = ga_side.get("repeat_stats")
    if rl_stats is None and ga_stats is None:
        return None

    metric_specs: List[Tuple[str, str, bool]] = [
        ("loss", "Loss", True),
        ("p", metric_names[0], False),
    ]
    if len(metric_names) > 1:
        metric_specs.append(("s", metric_names[1], False))
    metric_specs.append(("time_ms", "Time(ms)", True))

    summary_rows = []
    for key, label, lower_better in metric_specs:
        rl_mean = rl_stats.get(f"{key}_mean") if rl_stats else None
        ga_mean = ga_stats.get(f"{key}_mean") if ga_stats else None
        if rl_mean is None and ga_mean is None:
            continue

        rl_mean = float(rl_mean) if rl_mean is not None else None
        ga_mean = float(ga_mean) if ga_mean is not None else None
        winner = "tie"
        if rl_mean is not None and ga_mean is not None:
            if abs(rl_mean - ga_mean) <= 1e-12:
                winner = "tie"
            elif lower_better:
                winner = "rl" if rl_mean < ga_mean else "ga"
            else:
                winner = "rl" if rl_mean > ga_mean else "ga"

        summary_rows.append(
            {
                "key": key,
                "label": label,
                "lower_better": lower_better,
                "winner": winner,
                "mean_gap_rl_minus_ga": (
                    float(rl_mean - ga_mean)
                    if rl_mean is not None and ga_mean is not None
                    else None
                ),
                "rl": {
                    "n": int(rl_stats.get("n", 0)) if rl_stats else 0,
                    "mean": rl_mean,
                    "std": float(rl_stats.get(f"{key}_std")) if rl_stats and rl_stats.get(f"{key}_std") is not None else None,
                    "var": _variance_from_stats(rl_stats, key),
                    "min": float(rl_stats.get(f"{key}_min")) if rl_stats and rl_stats.get(f"{key}_min") is not None else None,
                    "max": float(rl_stats.get(f"{key}_max")) if rl_stats and rl_stats.get(f"{key}_max") is not None else None,
                },
                "ga": {
                    "n": int(ga_stats.get("n", 0)) if ga_stats else 0,
                    "mean": ga_mean,
                    "std": float(ga_stats.get(f"{key}_std")) if ga_stats and ga_stats.get(f"{key}_std") is not None else None,
                    "var": _variance_from_stats(ga_stats, key),
                    "min": float(ga_stats.get(f"{key}_min")) if ga_stats and ga_stats.get(f"{key}_min") is not None else None,
                    "max": float(ga_stats.get(f"{key}_max")) if ga_stats and ga_stats.get(f"{key}_max") is not None else None,
                },
            }
        )

    return {
        "rl_repeat_count": int(rl_stats.get("n", 0)) if rl_stats else 0,
        "ga_repeat_count": int(ga_stats.get("n", 0)) if ga_stats else 0,
        "metrics": summary_rows,
    }


def build_compare_evaluator(
    *,
    base_model: str,
    data_path: str,
    batch_size: int,
    run_output_dir: str,
    model_type: Optional[str] = None,
    search_algorithm: str,
    stage1_rl_lr: Optional[str],
    stage2_rl_lr: Optional[str],
    random_seed: int,
    perm_trials: int,
    cost_trials: int,
    budget_trials: int,
    final_eval_repeat_n: int,
    final_eval_config_source: str = "json",
    final_eval_config_path: str = "glue_final_configs_best_ppo.json",
    skip_stage1_rl: bool = True,
    skip_noise_rl: bool = True,
    skip_final_eval: bool = True,
    stage2_k_trials: Optional[int] = None,
    stage2_probe_size: Optional[int] = None,
):
    seed_everything(random_seed)

    from datasets import load_dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        LlamaTokenizer,
    )

    from layer_importance_evaluator import LayerImportanceEvaluator

    if "llama" in base_model and "llama3" not in base_model:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"

    dataset_key = data_path.lower()
    if dataset_key == "stsb":
        num_labels = 1
    elif dataset_key == "mnli":
        num_labels = 3
    else:
        num_labels = 2

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=num_labels,
        trust_remote_code=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def tokenize_fn(examples):
        dp = dataset_key
        if dp in ("sst2", "cola"):
            return tokenizer(
                examples["sentence"],
                truncation=True,
                padding=False,
                max_length=128,
                return_tensors=None,
            )
        if dp == "qnli":
            return tokenizer(
                examples["question"],
                examples["sentence"],
                truncation=True,
                padding=False,
                max_length=128,
                return_tensors=None,
            )
        if dp == "mnli":
            return tokenizer(
                examples["premise"],
                examples["hypothesis"],
                truncation=True,
                padding=False,
                max_length=128,
                return_tensors=None,
            )
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            padding=False,
            max_length=128,
            return_tensors=None,
        )

    data = load_dataset("nyu-mll/glue", data_path)
    val_data_mm = None
    if dataset_key == "mnli":
        train_data = data["train"].shuffle(seed=random_seed).map(tokenize_fn)
        val_data = data["validation_matched"].shuffle(seed=random_seed).map(tokenize_fn)
        val_data_mm = data["validation_mismatched"].shuffle(seed=random_seed).map(tokenize_fn)
        train_data = train_data.rename_column("label", "labels")
        val_data = val_data.rename_column("label", "labels")
        val_data_mm = val_data_mm.rename_column("label", "labels")
        mm_columns = [c for c in ("input_ids", "attention_mask", "token_type_ids", "labels") if c in val_data_mm.column_names]
        val_data_mm.set_format(type="torch", columns=mm_columns)
    else:
        train_data = data["train"].shuffle(seed=random_seed).map(tokenize_fn)
        val_data = data["validation"].shuffle(seed=random_seed).map(tokenize_fn)
        train_data = train_data.rename_column("label", "labels")
        val_data = val_data.rename_column("label", "labels")

    columns = [c for c in ("input_ids", "attention_mask", "token_type_ids", "labels") if c in train_data.column_names]
    train_data.set_format(type="torch", columns=columns)
    val_data.set_format(type="torch", columns=[c for c in columns if c in val_data.column_names])

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
        pad_to_multiple_of=8,
    )

    evaluator = LayerImportanceEvaluator(
        model=model,
        train_data=train_data,
        test_data=val_data,
        test_data_mm=val_data_mm,
        data_collator=data_collator,
        batch_size=batch_size,
        stage1_rl_lr=stage1_rl_lr,
        stage2_rl_lr=stage2_rl_lr,
        run_output_dir=run_output_dir,
        final_eval_config_source=final_eval_config_source,
        final_eval_config_path=final_eval_config_path,
        final_eval_random_seed=random_seed,
        final_eval_permutation_trials=0,
        final_eval_cost_equivalent_trials=0,
        final_eval_budget_equivalent_trials=0,
        final_eval_stage1_budget_trials=0,
        final_eval_stage2_budget_trials=0,
        final_eval_repeat_n=final_eval_repeat_n,
        skip_stage1_rl=skip_stage1_rl,
        skip_noise_rl=skip_noise_rl,
        skip_final_eval=skip_final_eval,
        data_path=data_path,
        search_algorithm=search_algorithm,
        stage2_k_trials=stage2_k_trials,
        stage2_probe_size=stage2_probe_size,
    )
    if model_type is not None:
        evaluator.model_type = normalize_model_type(model_type)
    return evaluator


def ensure_final_eval_json(
    *,
    algorithm: str,
    run_dir: Path,
    side_config: CompareSideConfig,
    dataset: str,
    base_model: str,
    data_path: str,
    batch_size: int,
    stage1_rl_lr: Optional[str],
    stage2_rl_lr: Optional[str],
    random_seed: int,
    perm_trials: int,
    cost_trials: int,
    budget_trials: int,
    final_eval_repeat_n: int,
    model_type: Optional[str] = None,
    prepared_evaluator=None,
) -> Tuple[Path, List[str]]:
    json_path = final_eval_json(run_dir, dataset)
    warnings: List[str] = []
    if json_path.is_file():
        existing = read_json(json_path)
        if final_eval_json_matches_protocol(
            existing,
            final_eval_repeat_n,
            expect_random_groups=False,
        ):
            return json_path, []
        warnings.append(
            f"{algorithm.upper()} 的 final-eval 文件使用旧评测协议或 repeat_n 不匹配，"
            "已重新生成统一最终评估。"
        )

    config_source = side_config.final_eval_config_source
    config_path = side_config.final_eval_config_path
    stage1_search_best = None
    stage2_search_best = None
    stage1_source = ""
    stage2_source = ""

    if config_source == "search":
        warnings.append(
            f"{algorithm.upper()} final-eval JSON missing/stale; "
            "regenerating unified final eval from current search results."
        )
        stage1_search_best, stage1_source = recover_stage1_search_best(run_dir, algorithm)
        stage2_search_best, stage2_source = recover_stage2_search_best(run_dir, algorithm)
    else:
        warnings.append(
            f"{algorithm.upper()} final-eval JSON missing/stale; "
            f"regenerating unified final eval from declared {config_source} config."
        )

    evaluator = prepared_evaluator
    created_evaluator = evaluator is None
    if evaluator is None:
        effective_config_path = config_path or default_final_eval_json_for_algorithm(algorithm)
        evaluator = build_compare_evaluator(
            base_model=base_model,
            data_path=data_path,
            batch_size=batch_size,
            run_output_dir=str(run_dir),
            model_type=model_type,
            search_algorithm=algorithm,
            stage1_rl_lr=stage1_rl_lr,
            stage2_rl_lr=stage2_rl_lr,
            random_seed=random_seed,
            perm_trials=perm_trials,
            cost_trials=cost_trials,
            budget_trials=budget_trials,
            final_eval_repeat_n=final_eval_repeat_n,
            final_eval_config_source=config_source,
            final_eval_config_path=effective_config_path,
            skip_stage1_rl=True,
            skip_noise_rl=True,
            skip_final_eval=False,
        )

    compare_perm_trials = 0
    compare_cost_trials = 0
    compare_budget_trials = 0
    compare_stage1_budget_trials = 0
    compare_stage2_budget_trials = 0

    from genetic_search_module import (
        GeneticUnifiedFinalEvaluationModule,
        build_stage1_context,
        build_stage2_context,
        build_stage2_final_eval_context_without_search,
        resolve_stage1_selected_config,
    )
    from final_evaluation_module import UnifiedFinalEvaluationModule

    stage1_context = build_stage1_context(
        evaluator, log_fn=evaluator.log, include_distribution=False,
        constraint_ratio=getattr(evaluator, "error_threshold", None),
    )
    if config_source == "search" and stage1_search_best is None:
        warnings.append(
            f"{algorithm.upper()} 未找到 Stage-1 checkpoint/search 结果，已回退到 baseline 配置生成对比结果。"
        )
        stage1_search_best = {
            "gelu": stage1_context.base_gelu.copy(),
            "softmax": stage1_context.base_softmax.copy(),
            "cost": float(stage1_context.base_tot_c),
        }
    elif config_source == "search":
        warnings.append(f"{algorithm.upper()} Stage-1 fallback 来源：{stage1_source}")

    fixed_gelu, fixed_softmax, fixed_label, fixed_resolved_source = resolve_stage1_selected_config(
        evaluator=evaluator,
        search_best_config=stage1_search_best,
        config_source=config_source,
        config_path=config_path,
    )

    if hasattr(evaluator, "activate_noise_logging"):
        previous_log_file = evaluator.activate_noise_logging()
        restore_log = lambda: evaluator.restore_log_file(previous_log_file)
    else:
        previous_log_file = getattr(evaluator, "active_log_file", None)
        noise_log_file = getattr(evaluator, "noise_log_file", None)
        if noise_log_file:
            noise_log_path = Path(noise_log_file)
            noise_log_path.parent.mkdir(parents=True, exist_ok=True)
            evaluator.active_log_file = str(noise_log_path)

        def restore_log():
            if previous_log_file is not None:
                evaluator.active_log_file = previous_log_file

    try:
        if stage2_search_best is not None:
            stage2_context = build_stage2_context(
                evaluator,
                fixed_gelu,
                fixed_softmax,
                log_fn=evaluator.log,
                limit_tolerance=getattr(evaluator, "stage2_limit_tolerance", None),
                stability_tolerance=getattr(evaluator, "stage2_stability_tolerance", None),
            )
            baseline_noise_tot_c = stage2_context.cost_reference_tot_c
            limit_loss = stage2_context.search_limits["loss"]
            limit_p = stage2_context.search_limits["metric1"]
            limit_s = stage2_context.search_limits["metric2"]
        else:
            stage2_eval_context = build_stage2_final_eval_context_without_search(evaluator)
            baseline_noise_tot_c = stage2_eval_context["baseline_tot_c"]
            limit_loss = stage2_eval_context["limit_loss"]
            limit_p = stage2_eval_context["limit_p"]
            limit_s = stage2_eval_context["limit_s"]
            if config_source == "search":
                warnings.append(
                    f"{algorithm.upper()} 未找到 Stage-2 稳定最优噪声配置，将仅输出 baseline/no-noise 对照并附带警告。"
                )

        module_cls = (
            GeneticUnifiedFinalEvaluationModule
            if algorithm == "ga"
            else UnifiedFinalEvaluationModule
        )
        runner = module_cls(
            evaluator=evaluator,
            config_source=config_source,
            config_path=config_path,
            random_seed=random_seed,
            permutation_trials=compare_perm_trials,
            cost_equivalent_trials=compare_cost_trials,
            budget_equivalent_trials=compare_budget_trials,
            stage1_budget_trials=compare_stage1_budget_trials,
            stage2_budget_trials=compare_stage2_budget_trials,
            repeat_n=final_eval_repeat_n,
            results_dir=evaluator.final_eval_dir,
        )
        result = runner.run(
            search_best_stage1=stage1_search_best,
            search_best_stage2=stage2_search_best,
            baseline_stage1_gelu=stage1_context.base_gelu,
            baseline_stage1_softmax=stage1_context.base_softmax,
            baseline_noise_tot_c=baseline_noise_tot_c,
            limit_loss=limit_loss,
            limit_p=limit_p,
            limit_s=limit_s,
        )
    finally:
        restore_log()
    summary_path = Path(result["summary_path"])
    if created_evaluator:
        release_compare_evaluator(evaluator)
    return summary_path, warnings


def format_pct_delta(selected: float, baseline: float, lower_better: bool = False) -> str:
    denom = baseline if abs(baseline) > 1e-12 else 1.0
    delta = (selected - baseline) / denom * 100.0
    if lower_better:
        return f"{delta:.2f}%（越低越好）"
    return f"{delta:.2f}%"


def build_stage_compare_payload(
    *,
    stage_label: str,
    dataset: str,
    compare_root: Path,
    rl_run_dir: Path,
    ga_run_dir: Path,
    rl_json_path: Path,
    ga_json_path: Path,
    rl_warnings: List[str],
    ga_warnings: List[str],
    process_meta: dict,
) -> dict:
    metric_names = DATASET_METRIC_SHORT_NAMES.get(dataset, ["Metric1", "Metric2"])
    rl_obj = read_json(rl_json_path) or {}
    ga_obj = read_json(ga_json_path) or {}

    def extract_side(
        label: str,
        algorithm: str,
        run_dir: Path,
        obj: dict,
        source_path: Path,
    ) -> dict:
        baseline = obj.get("baseline") or {}
        selected = obj.get("optimized")
        selected_origin = "optimized"
        if selected is None:
            selected = obj.get("selected")
            selected_origin = "selected"
        selected_warning = None
        if selected is None:
            fallback_candidate = obj.get("no_noise") or baseline
            selected = fallback_candidate
            selected_origin = "no_noise" if obj.get("no_noise") is not None else "baseline"
            selected_warning = (
                "未获得正常的 selected 结果，当前展示已回退到 "
                f"{selected_origin}。"
            )
        process_state = (process_meta.get(algorithm) or {}).get("state", "-")
        process_return_code = (process_meta.get(algorithm) or {}).get("return_code")
        stage1_selected_config = obj.get("fixed_stage1_config")
        stage1_selected_source = "final_eval_fixed_stage1_config"
        if stage1_selected_config is None:
            optimized_stage1 = obj.get("optimized_stage1") or {}
            if optimized_stage1.get("gelu") is not None and optimized_stage1.get("softmax") is not None:
                stage1_selected_config = {
                    "gelu": optimized_stage1["gelu"],
                    "softmax": optimized_stage1["softmax"],
                }
                stage1_selected_source = "final_eval_optimized_stage1"
        if stage1_selected_config is None and selected is not None:
            if selected.get("gelu") is not None and selected.get("softmax") is not None:
                stage1_selected_config = {
                    "gelu": selected["gelu"],
                    "softmax": selected["softmax"],
                }
                stage1_selected_source = f"final_eval_{selected_origin}"
        if stage1_selected_config is None:
            stage1_selected_config, stage1_selected_source = (
                resolve_stage1_selected_config_from_artifacts(run_dir, algorithm, dataset)
            )
        repeat_stats, repeat_trials = _extract_repeat_evaluation(obj)
        return {
            "label": label,
            "baseline": baseline,
            "selected": selected,
            "selected_single": obj.get("selected_single"),
            "selected_origin": selected_origin,
            "selected_warning": selected_warning,
            "json_path": str(source_path),
            "status": obj.get("status", "ok"),
            "message": obj.get("message"),
            "selected_source": obj.get("selected_source"),
            "process_state": process_state,
            "process_return_code": process_return_code,
            "stage1_selected_config": stage1_selected_config,
            "stage1_selected_source": stage1_selected_source,
            "repeat_stats": repeat_stats,
            "repeat_trials": repeat_trials,
            "baseline_repeat_stats": (obj.get("baseline_repeat_evaluation") or {}).get("stats"),
        }

    rl_side = extract_side("RL", "rl", rl_run_dir, rl_obj, rl_json_path)
    ga_side = extract_side("GA", "ga", ga_run_dir, ga_obj, ga_json_path)

    payload = {
        "stage": stage_label,
        "dataset": dataset,
        "generated_at": now_ts(),
        "compare_root": str(compare_root),
        "rl_run_dir": str(rl_run_dir),
        "ga_run_dir": str(ga_run_dir),
        "metric_names": metric_names,
        "process_meta": process_meta,
        "warnings": [],
        "sides": {
            "rl": rl_side,
            "ga": ga_side,
        },
    }

    payload["warnings"].extend(process_meta.get("compare_warnings", []))
    payload["warnings"].extend(rl_warnings)
    payload["warnings"].extend(ga_warnings)
    if rl_side["selected_warning"]:
        payload["warnings"].append(f"RL: {rl_side['selected_warning']}")
    if ga_side["selected_warning"]:
        payload["warnings"].append(f"GA: {ga_side['selected_warning']}")
    if rl_side.get("message"):
        payload["warnings"].append(f"RL 结果说明：{rl_side['message']}")
    if ga_side.get("message"):
        payload["warnings"].append(f"GA 结果说明：{ga_side['message']}")
    for label, side in (("RL", rl_side), ("GA", ga_side)):
        state = side.get("process_state")
        if state not in ("running", "completed", "-", None):
            payload["warnings"].append(
                f"{label} 进程状态为 {state}；本次对比结果可能基于中断前保存的当前最优配置。"
            )
    for label, side in (("RL", rl_side), ("GA", ga_side)):
        if stage_label in ("stage2", "final") and not side.get("stage1_selected_config"):
            payload["warnings"].append(
                f"{label} Stage-2 对比未能解析出固定的 Stage-1 配置；当前结果可能已回退到 baseline。"
            )
    if stage_label in ("stage2", "final"):
        payload["stage2_repeat_summary"] = _build_stage2_repeat_summary(
            metric_names,
            rl_side,
            ga_side,
        )
    return payload


def save_stage_compare_report(payload: dict, output_dir: Path) -> Tuple[Path, Path]:
    ensure_dir(output_dir)
    source_stage = payload["stage"]
    stage = "final" if source_stage in ("stage1", "stage2", "final") else source_stage
    output_payload = dict(payload)
    output_payload["stage"] = stage
    dataset = payload["dataset"]
    metric_names = payload["metric_names"]
    rl_side = payload["sides"]["rl"]
    ga_side = payload["sides"]["ga"]
    json_path = output_dir / f"{stage}_compare_summary_{dataset}.json"
    md_path = output_dir / f"{stage}_compare_report_{dataset}.md"
    plot_path = output_dir / f"{stage}_compare_plot_{dataset}.png"
    if stage == "final":
        for legacy_stage in ("stage1", "stage2"):
            for legacy_path in output_dir.glob(f"{legacy_stage}_compare_*_{dataset}.*"):
                try:
                    legacy_path.unlink()
                except OSError:
                    pass

    write_json(json_path, output_payload)

    def stage1_cost_value(result: dict) -> Optional[float]:
        if result.get("stage1_tot_c") is not None:
            return float(result["stage1_tot_c"])
        if result.get("tot_c") is not None and result.get("noise_config") is None:
            return float(result["tot_c"])
        return None

    def stage2_cost_value(result: dict) -> Optional[float]:
        if result.get("stage2_tot_c") is not None:
            return float(result["stage2_tot_c"])
        if result.get("tot_c") is not None and (
            result.get("noise_config") is not None or result.get("breakdown") is not None
        ):
            return float(result["tot_c"])
        return None

    def total_cost_value(result: dict) -> Optional[float]:
        s1_cost = stage1_cost_value(result)
        s2_cost = stage2_cost_value(result)
        if s1_cost is not None and s2_cost is not None:
            return s1_cost + s2_cost
        if result.get("tot_c") is not None:
            return float(result["tot_c"])
        return s1_cost if s1_cost is not None else s2_cost

    def format_cost(value: Optional[float]) -> str:
        return f"{float(value):.4f}" if value is not None else "-"

    def row(label: str, side: dict) -> List[str]:
        selected = side["selected"] or {}
        baseline = side["baseline"] or {}
        metric2_text = "-"
        if "s" in selected and selected.get("s") is not None:
            metric2_text = f"{float(selected['s']):.6f}"
        delta_loss = "-"
        delta_m1 = "-"
        delta_m2 = "-"
        if baseline:
            delta_loss = format_pct_delta(float(selected.get("loss", 0.0)), float(baseline.get("loss", 0.0)), lower_better=True)
            delta_m1 = format_pct_delta(float(selected.get("p", 0.0)), float(baseline.get("p", 0.0)))
            if "s" in selected and baseline.get("s") is not None:
                delta_m2 = format_pct_delta(float(selected.get("s", 0.0)), float(baseline.get("s", 0.0)))
        return [
            label,
            side.get("status") or "-",
            side.get("process_state") or "-",
            side.get("selected_origin") or "-",
            side.get("selected_source") or "-",
            f"{float(selected.get('loss', 0.0)):.6f}",
            f"{float(selected.get('p', 0.0)):.6f}",
            metric2_text,
            format_cost(stage1_cost_value(selected)),
            format_cost(stage2_cost_value(selected)),
            format_cost(total_cost_value(selected)),
            f"{float(selected.get('time_ms', 0.0)):.3f}" if selected.get("time_ms") is not None else "-",
            "Y" if selected.get("feasible", False) else "N",
            delta_loss,
            delta_m1,
            delta_m2,
        ]

    header_metric2 = metric_names[1] if len(metric_names) > 1 else "-"
    header = [
        "算法",
        "评估状态",
        "进程状态",
        "展示来源",
        "配置来源",
        "Loss",
        metric_names[0],
        header_metric2,
        "Stage1 Cost",
        "Stage2 Cost",
        "Total Cost",
        "Time(ms)",
        "Feasible",
        "dLoss%",
        f"d{metric_names[0]}%",
        f"d{header_metric2}%" if len(metric_names) > 1 else "-",
    ]
    rows = [row("RL", rl_side), row("GA", ga_side)]
    stage2_repeat_summary = payload.get("stage2_repeat_summary")

    lines = [
        f"# {stage.upper()}：RL 与 GA 对比报告",
        "",
        f"- 数据集：`{dataset}`",
        f"- 生成时间：`{payload['generated_at']}`",
        f"- RL 运行目录：`{payload['rl_run_dir']}`",
        f"- GA 运行目录：`{payload['ga_run_dir']}`",
        "",
        "## 指标对比",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for values in rows:
        lines.append("| " + " | ".join(values) + " |")

    lines.extend([
        "",
        "## 关键配置",
        "",
        f"- RL 结果文件：`{rl_side['json_path']}`",
        f"- GA 结果文件：`{ga_side['json_path']}`",
        f"- RL 选中配置来源：`{rl_side.get('selected_source')}`",
        f"- GA 选中配置来源：`{ga_side.get('selected_source')}`",
    ])

    if stage == "stage1":
        lines.extend(
            [
                f"- RL GELU：`{rl_side['selected'].get('gelu')}`",
                f"- RL Softmax：`{rl_side['selected'].get('softmax')}`",
                f"- GA GELU：`{ga_side['selected'].get('gelu')}`",
                f"- GA Softmax：`{ga_side['selected'].get('softmax')}`",
            ]
        )
    else:
        rl_fixed_stage1 = rl_side.get("stage1_selected_config") or {}
        ga_fixed_stage1 = ga_side.get("stage1_selected_config") or {}
        rl_noise_cfg = (rl_side["selected"] or {}).get("noise_config")
        ga_noise_cfg = (ga_side["selected"] or {}).get("noise_config")
        rl_breakdown = (rl_side["selected"] or {}).get("stage2_breakdown") or (rl_side["selected"] or {}).get("breakdown")
        ga_breakdown = (ga_side["selected"] or {}).get("stage2_breakdown") or (ga_side["selected"] or {}).get("breakdown")
        lines.extend(
            [
                f"- RL 固定的 Stage-1 GELU：`{rl_fixed_stage1.get('gelu')}`",
                f"- RL 固定的 Stage-1 Softmax：`{rl_fixed_stage1.get('softmax')}`",
                f"- RL 固定配置来源：`{rl_side.get('stage1_selected_source')}`",
                f"- RL 选中噪声配置：`{rl_noise_cfg}`",
                f"- RL 噪声 cost breakdown：`{rl_breakdown}`",
                f"- GA 固定的 Stage-1 GELU：`{ga_fixed_stage1.get('gelu')}`",
                f"- GA 固定的 Stage-1 Softmax：`{ga_fixed_stage1.get('softmax')}`",
                f"- GA 固定配置来源：`{ga_side.get('stage1_selected_source')}`",
                f"- GA 选中噪声配置：`{ga_noise_cfg}`",
                f"- GA 噪声 cost breakdown：`{ga_breakdown}`",
            ]
        )

    if payload["warnings"]:
        lines.extend(["", "## 警告", ""])
        for item in payload["warnings"]:
            lines.append(f"- {item}")

    if stage != "stage1" and stage2_repeat_summary:
        lines.extend(
            [
                "",
                "## Stage-2 多次评估统计",
                "",
                f"- RL 重复评估次数：`{stage2_repeat_summary.get('rl_repeat_count', 0)}`",
                f"- GA 重复评估次数：`{stage2_repeat_summary.get('ga_repeat_count', 0)}`",
                "",
                "| 指标 | RL 均值 | RL 标准差 | RL 方差 | RL 最小值 | RL 最大值 | GA 均值 | GA 标准差 | GA 方差 | GA 最小值 | GA 最大值 | RL-GA 均值差 | 更优方 |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for metric_row in stage2_repeat_summary.get("metrics", []):
            rl_stats = metric_row.get("rl") or {}
            ga_stats = metric_row.get("ga") or {}
            winner_text = {"rl": "RL", "ga": "GA", "tie": "持平"}.get(
                metric_row.get("winner", "tie"),
                str(metric_row.get("winner", "tie")),
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(metric_row.get("label", "-")),
                        f"{float(rl_stats.get('mean', 0.0)):.6f}" if rl_stats.get("mean") is not None else "-",
                        f"{float(rl_stats.get('std', 0.0)):.6f}" if rl_stats.get("std") is not None else "-",
                        f"{float(rl_stats.get('var', 0.0)):.6f}" if rl_stats.get("var") is not None else "-",
                        f"{float(rl_stats.get('min', 0.0)):.6f}" if rl_stats.get("min") is not None else "-",
                        f"{float(rl_stats.get('max', 0.0)):.6f}" if rl_stats.get("max") is not None else "-",
                        f"{float(ga_stats.get('mean', 0.0)):.6f}" if ga_stats.get("mean") is not None else "-",
                        f"{float(ga_stats.get('std', 0.0)):.6f}" if ga_stats.get("std") is not None else "-",
                        f"{float(ga_stats.get('var', 0.0)):.6f}" if ga_stats.get("var") is not None else "-",
                        f"{float(ga_stats.get('min', 0.0)):.6f}" if ga_stats.get("min") is not None else "-",
                        f"{float(ga_stats.get('max', 0.0)):.6f}" if ga_stats.get("max") is not None else "-",
                        f"{float(metric_row.get('mean_gap_rl_minus_ga', 0.0)):.6f}" if metric_row.get("mean_gap_rl_minus_ga") is not None else "-",
                        winner_text,
                    ]
                )
                + " |"
            )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _plot_stage_compare(output_payload, plot_path)
    return md_path, plot_path


def _plot_stage_compare(payload: dict, plot_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log(f"[Warning] 对比图生成失败：{exc}")
        return

    stage = payload["stage"]
    metric_names = payload["metric_names"]
    rl_side = payload["sides"]["rl"]["selected"]
    ga_side = payload["sides"]["ga"]["selected"]
    stage2_repeat_summary = payload.get("stage2_repeat_summary")

    def _stage2_cost_for_plot(result: dict) -> float:
        if result.get("stage2_tot_c") is not None:
            return float(result.get("stage2_tot_c") or 0.0)
        if result.get("tot_c") is not None and (
            result.get("noise_config") is not None or result.get("breakdown") is not None
        ):
            return float(result.get("tot_c") or 0.0)
        return 0.0

    def _set_bar_ylim(ax, values: List[float]) -> None:
        finite_values = [abs(float(v)) for v in values if np.isfinite(float(v))]
        upper = max(finite_values) if finite_values else 0.0
        ax.set_ylim(0.0, upper * 1.22 if upper > 0.0 else 1.0)

    if stage in ("stage2", "final") and stage2_repeat_summary:
        summary_metrics = stage2_repeat_summary.get("metrics", [])
        display_rows = summary_metrics[:3]
        include_time = not any(item.get("key") == "time_ms" for item in display_rows)
        if include_time:
            time_metric = next(
                (item for item in summary_metrics if item.get("key") == "time_ms"),
                None,
            )
            if time_metric is not None and len(display_rows) < 3:
                display_rows.append(time_metric)

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle(f"{stage.upper()}: RL vs GA Repeated Evaluation", fontsize=14, fontweight="bold")
        colors = {"RL": "#4C78A8", "GA": "#E45756"}

        for ax, metric_row in zip(axes.flat[:3], display_rows):
            rl_stats = metric_row.get("rl") or {}
            ga_stats = metric_row.get("ga") or {}
            means = [
                float(rl_stats.get("mean", 0.0) or 0.0),
                float(ga_stats.get("mean", 0.0) or 0.0),
            ]
            stds = [
                float(rl_stats.get("std", 0.0) or 0.0),
                float(ga_stats.get("std", 0.0) or 0.0),
            ]
            mins = [rl_stats.get("min"), ga_stats.get("min")]
            maxs = [rl_stats.get("max"), ga_stats.get("max")]
            ax.bar(
                ["RL", "GA"],
                means,
                yerr=stds,
                capsize=6,
                color=[colors["RL"], colors["GA"]],
                alpha=0.85,
            )
            ax.set_title(f"{metric_row.get('label')} (mean ± std)")
            ax.grid(True, axis="y", alpha=0.3)
            for idx, value in enumerate(means):
                min_text = f"{float(mins[idx]):.4f}" if mins[idx] is not None else "-"
                max_text = f"{float(maxs[idx]):.4f}" if maxs[idx] is not None else "-"
                ax.text(
                    idx,
                    value,
                    f"{value:.4f}\nσ={stds[idx]:.4f}\n[{min_text}, {max_text}]",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        cost_ax = axes.flat[3]
        cost_values = [
            _stage2_cost_for_plot(rl_side),
            _stage2_cost_for_plot(ga_side),
        ]
        cost_ax.bar(["RL", "GA"], cost_values, color=[colors["RL"], colors["GA"]], alpha=0.85)
        cost_ax.set_title("Stage-2 Noise Cost")
        cost_ax.grid(True, axis="y", alpha=0.3)
        _set_bar_ylim(cost_ax, cost_values)
        for idx, value in enumerate(cost_values):
            cost_ax.annotate(
                f"{value:.4f}",
                xy=(idx, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        rl_n = stage2_repeat_summary.get("rl_repeat_count", 0)
        ga_n = stage2_repeat_summary.get("ga_repeat_count", 0)
        cost_ax.text(
            0.5,
            0.95,
            f"Repeat count: RL={rl_n}, GA={ga_n}",
            ha="center",
            va="top",
            transform=cost_ax.transAxes,
            fontsize=9,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(plot_path, dpi=180)
        plt.close(fig)
        return

    metrics = [
        ("loss", "Loss"),
        ("p", metric_names[0]),
    ]
    if len(metric_names) > 1:
        metrics.append(("s", metric_names[1]))
    else:
        metrics.append(("time_ms", "Time(ms)"))
    if any(side.get("stage2_tot_c") is not None for side in (rl_side, ga_side)):
        metrics.append(("stage2_tot_c", "Stage-2 Cost"))
    elif any(side.get("stage1_tot_c") is not None for side in (rl_side, ga_side)):
        metrics.append(("stage1_tot_c", "Stage-1 Cost"))
    else:
        metrics.append(("tot_c", "Cost"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{stage.upper()}: RL vs GA", fontsize=14, fontweight="bold")
    colors = {"RL": "#4C78A8", "GA": "#E45756"}

    for ax, (key, label) in zip(axes.flat, metrics):
        rl_value = float(rl_side.get(key, 0.0) or 0.0)
        ga_value = float(ga_side.get(key, 0.0) or 0.0)
        ax.bar(["RL", "GA"], [rl_value, ga_value], color=[colors["RL"], colors["GA"]])
        ax.set_title(label)
        ax.grid(True, axis="y", alpha=0.3)
        if "Cost" in label:
            _set_bar_ylim(ax, [rl_value, ga_value])
        for idx, value in enumerate([rl_value, ga_value]):
            ax.annotate(
                f"{value:.4f}",
                xy=(idx, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(plot_path, dpi=180)
    plt.close(fig)


def build_child_command(
    *,
    python_exe: str,
    algorithm: str,
    side_config: CompareSideConfig,
    base_model: str,
    data_path: str,
    run_dir: Path,
    batch_size: int,
    stage1_search_episodes: int,
    stage2_search_episodes: int,
    stage1_search_generations: int,
    stage2_search_generations: int,
    stage1_search_lr: Optional[str],
    stage2_search_lr: Optional[str],
    random_seed: int,
    perm_trials: int,
    cost_trials: int,
    budget_trials: int,
    stage2_compare_repeats: int,
    stage1_accuracy_tolerance: Optional[float] = None,
    stage2_limit_tolerance: Optional[float] = None,
    stage2_stability_tolerance: Optional[float] = None,
    stage2_k_trials: Optional[int] = None,
    stage2_probe_size: Optional[int] = None,
) -> List[str]:
    entrypoint = "rl_tune.py" if algorithm == "rl" else "rl_tune_genetic.py"

    cmd = [
        python_exe,
        entrypoint,
        "--base_model", base_model,
        "--data_path", data_path,
        "--output_dir", str(run_dir),
        "--batch_size", str(batch_size),
        "--micro_batch_size", str(batch_size),
        "--num_epochs", "1",
        "--learning_rate", "2e-4",
        "--cutoff_len", "256",
        "--val_set_size", "120",
        "--eval_step", "80",
        "--adapter_name", "lora",
        "--target_modules", TARGET_MODULES_LITERAL,
        "--use_ist",
        "--final_eval_config_source", side_config.final_eval_config_source,
        "--final_eval_config_path", side_config.final_eval_config_path,
        "--manual_stage1_gelu", "",
        "--manual_stage1_softmax", "",
        "--manual_stage2_noise", "",
        "--final_eval_random_seed", str(random_seed),
        "--final_eval_permutation_trials", "0",
        "--final_eval_cost_equivalent_trials", "0",
        "--final_eval_budget_equivalent_trials", "0",
        "--final_eval_stage1_budget_trials", "0",
        "--final_eval_stage2_budget_trials", "0",
        "--final_eval_repeat_n", str(stage2_compare_repeats),
        "--skip_noise_rl", "true" if side_config.skip_noise_search else "false",
        "--skip_stage1_rl", "true" if side_config.skip_stage1_search else "false",
        "--skip_final_eval", "false",
        "--resume_run_dir", "",
    ]
    if algorithm == "rl":
        if not side_config.skip_stage1_search:
            cmd.extend(
                [
                    "--stage1_rl_episodes", str(stage1_search_episodes),
                    "--stage1_rl_episodes_specified", "true",
                ]
            )
        if not side_config.skip_noise_search:
            cmd.extend(
                [
                    "--stage2_rl_episodes", str(stage2_search_episodes),
                    "--stage2_rl_episodes_specified", "true",
                ]
            )
        cmd.extend(["--stage1_rl_lr", str(stage1_search_lr), "--stage2_rl_lr", str(stage2_search_lr)])
    else:
        if not side_config.skip_stage1_search:
            cmd.extend(
                [
                    "--stage1_ga_generations", str(stage1_search_generations),
                    "--stage1_ga_generations_specified", "true",
                ]
            )
        if not side_config.skip_noise_search:
            cmd.extend(
                [
                    "--stage2_ga_generations", str(stage2_search_generations),
                    "--stage2_ga_generations_specified", "true",
                ]
            )
    if stage1_accuracy_tolerance is not None:
        cmd.extend(["--stage1_accuracy_tolerance", str(stage1_accuracy_tolerance)])
    if stage2_limit_tolerance is not None:
        cmd.extend(["--stage2_limit_tolerance", str(stage2_limit_tolerance)])
    if stage2_stability_tolerance is not None:
        cmd.extend(["--stage2_stability_tolerance", str(stage2_stability_tolerance)])
    if stage2_k_trials is not None:
        cmd.extend(["--stage2_k_trials", str(int(stage2_k_trials))])
    if stage2_probe_size is not None:
        cmd.extend(["--stage2_probe_size", str(int(stage2_probe_size))])
    return cmd


def start_child(spec: ChildRunSpec, extra_env: Dict[str, str]) -> None:
    ensure_dir(spec.run_dir / "logs")
    ensure_dir(spec.log_path.parent)
    env = os.environ.copy()
    env.update(extra_env)
    env.update(spec.env_overrides)
    popen_kwargs = {
        "stdout": None,
        "stderr": subprocess.STDOUT,
        "cwd": os.getcwd(),
        "env": env,
        "start_new_session": True,
    }
    preexec_fn = _build_parent_death_preexec_fn()
    if preexec_fn is not None:
        popen_kwargs["preexec_fn"] = preexec_fn
    with spec.log_path.open("w", encoding="utf-8") as log_handle:
        popen_kwargs["stdout"] = log_handle
        spec.process = subprocess.Popen(
            spec.command,
            **popen_kwargs,
        )


def child_return_code(spec: ChildRunSpec) -> Optional[int]:
    if spec.process is None:
        return None
    return spec.process.poll()


def send_interrupt(spec: ChildRunSpec) -> None:
    if spec.process is None:
        return
    if spec.process.poll() is not None:
        return
    try:
        spec.process.send_signal(signal.SIGINT)
    except Exception:
        try:
            spec.process.terminate()
        except Exception:
            pass


def summarize_process_state(
    spec: ChildRunSpec,
    dataset: str,
    global_stop_requested: bool,
) -> dict:
    rc = child_return_code(spec)
    final_ok = final_eval_json(spec.run_dir, dataset).is_file()
    if rc is None:
        state = "running"
    elif rc == 0 and final_ok:
        state = "completed"
    elif rc == 0 and global_stop_requested:
        state = "stopped_by_request"
    elif rc == 0:
        state = "exited_early"
    else:
        state = f"failed(rc={rc})"
    return {
        "algorithm": spec.algorithm,
        "pid": spec.process.pid if spec.process is not None else None,
        "return_code": rc,
        "state": state,
        "final_eval_ready": final_ok,
        "run_dir": str(spec.run_dir),
        "log_path": str(spec.log_path),
        "command": spec.command,
        "env_overrides": spec.env_overrides,
    }


def child_error_summary_path(spec: ChildRunSpec) -> Path:
    return spec.run_dir / "logs" / "error_summary.txt"


def is_abnormal_final_state(state: dict) -> bool:
    return state.get("state") not in ("completed", "stopped_by_request")


def build_compare_error_sections(
    rl_state: Optional[dict],
    ga_state: Optional[dict],
    rl_spec: ChildRunSpec,
    ga_spec: ChildRunSpec,
) -> List[Tuple[str, str]]:
    sections: List[Tuple[str, str]] = []
    state_pairs = (
        ("RL", rl_state, rl_spec),
        ("GA", ga_state, ga_spec),
    )
    for label, state, spec in state_pairs:
        if state:
            sections.append(
                (
                    f"{label} Process State",
                    json.dumps(to_jsonable(state), ensure_ascii=False, indent=2),
                )
            )
        summary_text = read_text_tail(child_error_summary_path(spec), max_lines=120)
        if summary_text:
            sections.append((f"{label} Child Error Summary", summary_text))
            continue
        if state and is_abnormal_final_state(state):
            log_tail = read_text_tail(spec.log_path, max_lines=80)
            if log_tail:
                sections.append((f"{label} Child Log Tail", log_tail))
    return sections


def classify_stage_json_input(
    *,
    path: Path,
    stage_label: str,
    dataset: str,
    model_type: str,
    expected_algorithm: str,
) -> str:
    if not path.is_file():
        raise CompareRunnerError(
            f"{expected_algorithm.upper()} {stage_label} JSON file does not exist: {path}"
        )
    inferred_family = infer_algorithm_family_from_path(path)
    if inferred_family not in ("unknown", expected_algorithm):
        raise CompareRunnerError(
            f"{expected_algorithm.upper()} {stage_label} JSON '{path}' looks like a {inferred_family.upper()} artifact."
        )

    obj = read_json(path)
    if not isinstance(obj, dict):
        raise CompareRunnerError(f"Invalid JSON file: {path}")

    if "selected" in obj or "baseline" in obj or "status" in obj:
        validate_stage_result_json(
            path,
            stage_label=stage_label,
            dataset=dataset,
            model_type=model_type,
        )
        return "result_json"

    if stage_label == "stage1":
        validate_stage1_config_template(
            path,
            dataset=dataset,
            model_type=model_type,
        )
    else:
        validate_stage2_config_template(
            path,
            dataset=dataset,
            model_type=model_type,
        )
    return "config_json"


def resolve_direct_side_spec(
    *,
    algorithm: str,
    dataset: str,
    model_type: str,
    run_dir: Path,
    stage1_json_path: str,
    stage2_json_path: str,
) -> EvaluationOnlySideSpec:
    stage1_path = Path(stage1_json_path).expanduser().resolve()
    stage2_path = Path(stage2_json_path).expanduser().resolve()
    stage1_kind = classify_stage_json_input(
        path=stage1_path,
        stage_label="stage1",
        dataset=dataset,
        model_type=model_type,
        expected_algorithm=algorithm,
    )
    stage2_kind = classify_stage_json_input(
        path=stage2_path,
        stage_label="stage2",
        dataset=dataset,
        model_type=model_type,
        expected_algorithm=algorithm,
    )
    any_json = stage1_kind == "config_json" or stage2_kind == "config_json"
    if stage1_kind == "config_json" and stage2_kind == "config_json" and stage1_path != stage2_path:
        raise CompareRunnerError(
            f"{algorithm.upper()} 直接比较时，Stage-1 和 Stage-2 的 JSON 配置应指向同一个合并文件，"
            f"但得到 {stage1_path} 与 {stage2_path}。"
        )
    merged_path = ""
    if stage2_kind == "config_json":
        merged_path = str(stage2_path)
    elif stage1_kind == "config_json":
        merged_path = str(stage1_path)
    side_config = normalize_compare_side_config(
        label=algorithm.upper(),
        skip_stage1_search=(stage1_kind == "config_json"),
        skip_noise_search=(stage2_kind == "config_json"),
        final_eval_config_source=("json" if any_json else "search"),
        final_eval_config_path=merged_path,
    )
    return EvaluationOnlySideSpec(
        algorithm=algorithm,
        run_dir=run_dir,
        side_config=side_config,
        stage1_input_kind=stage1_kind,
        stage2_input_kind=stage2_kind,
        stage1_input_path=stage1_path,
        stage2_input_path=stage2_path,
        source_metadata={
            "compare_config_mode": "direct",
            "stage1_json_path": str(stage1_path),
            "stage2_json_path": str(stage2_path),
        },
    )


def resolve_persistent_side_spec(
    *,
    algorithm: str,
    dataset: str,
    model_type: str,
    persistent_root: Path,
    stage1_accuracy_tolerance: float,
    stage2_limit_tolerance: float,
    stage2_stability_tolerance: float,
) -> EvaluationOnlySideSpec:
    run_dir = persistent_run_dir_for_compare(
        persistent_root=persistent_root,
        algorithm=algorithm,
        model_type=model_type,
        dataset=dataset,
        stage1_accuracy_tolerance=stage1_accuracy_tolerance,
        stage2_limit_tolerance=stage2_limit_tolerance,
        stage2_stability_tolerance=stage2_stability_tolerance,
    ).resolve()
    if not run_dir.is_dir():
        raise CompareRunnerError(
            f"Persistent compare directory not found for {algorithm.upper()}: {run_dir}"
        )
    metadata = validate_persistent_run_dir(
        run_dir=run_dir,
        algorithm=algorithm,
        model_type=model_type,
        dataset=dataset,
    )
    ensure_persistent_side_has_compare_artifacts(
        run_dir=run_dir,
        algorithm=algorithm,
        dataset=dataset,
    )
    return EvaluationOnlySideSpec(
        algorithm=algorithm,
        run_dir=run_dir,
        side_config=CompareSideConfig(),
        stage1_input_kind="persistent_artifact",
        stage2_input_kind="persistent_artifact",
        source_metadata={
            "compare_config_mode": "persistent",
            "persistent_run_dir": str(run_dir),
            "constraint_slug": compare_constraint_slug(
                stage1_accuracy_tolerance,
                stage2_limit_tolerance,
                stage2_stability_tolerance,
            ),
            "metadata": metadata,
        },
    )


def materialize_direct_result_jsons(side_spec: EvaluationOnlySideSpec, *, dataset: str) -> None:
    if side_spec.stage1_input_kind == "result_json" and side_spec.stage1_input_path is not None:
        materialize_result_json(
            stage_label="stage1",
            src_path=side_spec.stage1_input_path,
            run_dir=side_spec.run_dir,
            dataset=dataset,
        )
    if side_spec.stage2_input_kind == "result_json" and side_spec.stage2_input_path is not None:
        materialize_result_json(
            stage_label="stage2",
            src_path=side_spec.stage2_input_path,
            run_dir=side_spec.run_dir,
            dataset=dataset,
        )


def side_needs_evaluator(side_spec: EvaluationOnlySideSpec, *, dataset: str) -> bool:
    return not final_eval_json(side_spec.run_dir, dataset).is_file()


def build_evaluation_only_process_state(
    side_spec: EvaluationOnlySideSpec,
    *,
    dataset: str,
) -> dict:
    return {
        "algorithm": side_spec.algorithm,
        "pid": None,
        "return_code": 0,
        "state": "completed",
        "final_eval_ready": final_eval_json(side_spec.run_dir, dataset).is_file(),
        "run_dir": str(side_spec.run_dir),
        "log_path": str(side_spec.run_dir / "logs" / "output.log"),
        "command": None,
        "env_overrides": {},
        "mode": "evaluation_only",
        "stage1_input_kind": side_spec.stage1_input_kind,
        "stage2_input_kind": side_spec.stage2_input_kind,
        "source_metadata": to_jsonable(side_spec.source_metadata or {}),
    }


def run_evaluation_only_compare(args: argparse.Namespace) -> int:
    compare_root = Path(args.output_dir).resolve()
    meta_dir = compare_root / "meta"
    children_dir = compare_root / "children"
    compare_dir = compare_root / "reports"
    compare_metadata_path = meta_dir / "compare_metadata.json"
    compare_runtime_path = meta_dir / "compare_runtime.json"
    compare_status_path = meta_dir / "compare_status.json"
    compare_final_status_path = meta_dir / "compare_final_status.json"
    compare_pid_path = meta_dir / "compare.pid"
    clear_error_summary(str(compare_root))

    model_type = normalize_model_type(args.model_type)
    dataset = str(args.dataset or "").strip().lower()
    compare_warnings: List[str] = []

    try:
        if args.compare_config_mode == "direct":
            rl_side_spec = resolve_direct_side_spec(
                algorithm="rl",
                dataset=dataset,
                model_type=model_type,
                run_dir=children_dir / "rl",
                stage1_json_path=args.rl_compare_stage1_json,
                stage2_json_path=args.rl_compare_stage2_json,
            )
            ga_side_spec = resolve_direct_side_spec(
                algorithm="ga",
                dataset=dataset,
                model_type=model_type,
                run_dir=children_dir / "ga",
                stage1_json_path=args.ga_compare_stage1_json,
                stage2_json_path=args.ga_compare_stage2_json,
            )
            materialize_direct_result_jsons(rl_side_spec, dataset=dataset)
            materialize_direct_result_jsons(ga_side_spec, dataset=dataset)
        elif args.compare_config_mode == "persistent":
            persistent_root = Path(args.compare_persistent_root).expanduser().resolve()
            rl_side_spec = resolve_persistent_side_spec(
                algorithm="rl",
                dataset=dataset,
                model_type=model_type,
                persistent_root=persistent_root,
                stage1_accuracy_tolerance=float(args.rl_compare_stage1_accuracy_tolerance),
                stage2_limit_tolerance=float(args.rl_compare_stage2_limit_tolerance),
                stage2_stability_tolerance=float(args.rl_compare_stage2_stability_tolerance),
            )
            ga_side_spec = resolve_persistent_side_spec(
                algorithm="ga",
                dataset=dataset,
                model_type=model_type,
                persistent_root=persistent_root,
                stage1_accuracy_tolerance=float(args.ga_compare_stage1_accuracy_tolerance),
                stage2_limit_tolerance=float(args.ga_compare_stage2_limit_tolerance),
                stage2_stability_tolerance=float(args.ga_compare_stage2_stability_tolerance),
            )
        else:
            raise CompareRunnerError(
                f"Unsupported compare_config_mode={args.compare_config_mode!r}."
            )

        metadata = {
            "dataset": dataset,
            "model_type": model_type,
            "base_model": args.base_model,
            "data_path": args.data_path,
            "compare_root": str(compare_root),
            "compare_config_mode": args.compare_config_mode,
            "stage2_compare_repeats": args.stage2_compare_repeats,
            "random_seed": args.random_seed,
            "perm_trials": args.perm_trials,
            "cost_trials": args.cost_trials,
            "budget_trials": args.budget_trials,
            "rl_side": to_jsonable(rl_side_spec.source_metadata or {}),
            "ga_side": to_jsonable(ga_side_spec.source_metadata or {}),
            "warnings": compare_warnings,
        }
        write_json(compare_metadata_path, metadata)
        if args.dry_run:
            log("dry-run 模式：仅写入 compare 元信息，不执行评估。")
            return 0

        write_json(
            compare_runtime_path,
            {
                "compare_pid": os.getpid(),
                "compare_config_mode": args.compare_config_mode,
            },
        )
        compare_pid_path.parent.mkdir(parents=True, exist_ok=True)
        compare_pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
        write_json(
            compare_status_path,
            {
                "updated_at": now_ts(),
                "mode": "evaluation_only",
                "compare_config_mode": args.compare_config_mode,
                "stage": "preparing",
            },
        )

        def _ensure_side_final_eval(
            *,
            algorithm: str,
            side_spec: EvaluationOnlySideSpec,
        ) -> Tuple[Path, List[str]]:
            needs_evaluator = side_needs_evaluator(side_spec, dataset=dataset)
            write_json(
                compare_status_path,
                {
                    "updated_at": now_ts(),
                    "mode": "evaluation_only",
                    "compare_config_mode": args.compare_config_mode,
                    "stage": f"final_eval_{algorithm}",
                    "algorithm": algorithm,
                    "needs_evaluator": needs_evaluator,
                },
            )
            if needs_evaluator:
                log(f"{algorithm.upper()} final-eval missing; running it before the other side to keep GPU memory bounded.")
            try:
                return ensure_final_eval_json(
                    algorithm=algorithm,
                    run_dir=side_spec.run_dir,
                    side_config=side_spec.side_config,
                    dataset=dataset,
                    base_model=args.base_model,
                    data_path=args.data_path,
                    batch_size=args.batch_size,
                    stage1_rl_lr=args.stage1_search_lr,
                    stage2_rl_lr=args.stage2_search_lr,
                    random_seed=args.random_seed,
                    perm_trials=args.perm_trials,
                    cost_trials=args.cost_trials,
                    budget_trials=args.budget_trials,
                    final_eval_repeat_n=args.stage2_compare_repeats,
                    model_type=model_type,
                )
            finally:
                cleanup_cuda_memory()

        rl_final_json, rl_final_warn = _ensure_side_final_eval(
            algorithm="rl",
            side_spec=rl_side_spec,
        )
        ga_final_json, ga_final_warn = _ensure_side_final_eval(
            algorithm="ga",
            side_spec=ga_side_spec,
        )
        rl_stage1_json = rl_stage2_json = rl_final_json
        ga_stage1_json = ga_stage2_json = ga_final_json
        rl_stage1_warn = rl_stage2_warn = rl_final_warn
        ga_stage1_warn = ga_stage2_warn = ga_final_warn

        rl_state = build_evaluation_only_process_state(rl_side_spec, dataset=dataset)
        ga_state = build_evaluation_only_process_state(ga_side_spec, dataset=dataset)
        stage1_payload = build_stage_compare_payload(
            stage_label="stage1",
            dataset=dataset,
            compare_root=compare_root,
            rl_run_dir=rl_side_spec.run_dir,
            ga_run_dir=ga_side_spec.run_dir,
            rl_json_path=rl_stage1_json,
            ga_json_path=ga_stage1_json,
            rl_warnings=rl_stage1_warn,
            ga_warnings=ga_stage1_warn,
            process_meta={
                "compare_warnings": compare_warnings,
                "rl": rl_state,
                "ga": ga_state,
            },
        )
        stage1_report_path, _ = save_stage_compare_report(stage1_payload, compare_dir)
        log(f"Stage-1 对比结果已生成：{stage1_report_path}")

        write_json(
            compare_status_path,
            {
                "updated_at": now_ts(),
                "mode": "evaluation_only",
                "compare_config_mode": args.compare_config_mode,
                "stage": "stage2",
                "stage1_report_path": str(stage1_report_path),
            },
        )

        rl_state = build_evaluation_only_process_state(rl_side_spec, dataset=dataset)
        ga_state = build_evaluation_only_process_state(ga_side_spec, dataset=dataset)
        stage2_payload = build_stage_compare_payload(
            stage_label="stage2",
            dataset=dataset,
            compare_root=compare_root,
            rl_run_dir=rl_side_spec.run_dir,
            ga_run_dir=ga_side_spec.run_dir,
            rl_json_path=rl_stage2_json,
            ga_json_path=ga_stage2_json,
            rl_warnings=rl_stage2_warn,
            ga_warnings=ga_stage2_warn,
            process_meta={
                "compare_warnings": compare_warnings,
                "rl": rl_state,
                "ga": ga_state,
            },
        )
        stage2_report_path, _ = save_stage_compare_report(stage2_payload, compare_dir)
        log(f"Stage-2 对比结果已生成：{stage2_report_path}")

        final_state = {
            "updated_at": now_ts(),
            "mode": "evaluation_only",
            "compare_config_mode": args.compare_config_mode,
            "rl": rl_state,
            "ga": ga_state,
            "final_report_path": str(stage2_report_path),
            "stage1_report_path": str(stage1_report_path),
            "stage2_report_path": str(stage2_report_path),
        }
        write_json(compare_final_status_path, final_state)
        write_json(compare_status_path, final_state)
        return 0
    except Exception as exc:
        error_payload = {
            "updated_at": now_ts(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(meta_dir / "compare_error.json", error_payload)
        write_error_summary(
            str(compare_root),
            program_name="rl_ga_compare_runner.py",
            status="failed",
            message=f"{type(exc).__name__}: {exc}",
            argv=sys.argv,
            exit_code=1,
            traceback_text=traceback.format_exc(),
        )
        log(f"[Error] 对比实验失败：{exc}")
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="并行运行 RL 与 GA，并生成阶段对比结果。")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument(
        "--compare-config-mode",
        choices=("direct", "persistent"),
        default="direct",
    )
    parser.add_argument("--compare-persistent-root", default="rl_results/persistent")
    parser.add_argument("--rl-compare-stage1-json", default="")
    parser.add_argument("--rl-compare-stage2-json", default="")
    parser.add_argument("--ga-compare-stage1-json", default="")
    parser.add_argument("--ga-compare-stage2-json", default="")
    parser.add_argument("--rl-compare-stage1-accuracy-tolerance", type=float, default=None)
    parser.add_argument("--rl-compare-stage2-limit-tolerance", type=float, default=None)
    parser.add_argument("--rl-compare-stage2-stability-tolerance", type=float, default=None)
    parser.add_argument("--ga-compare-stage1-accuracy-tolerance", type=float, default=None)
    parser.add_argument("--ga-compare-stage2-limit-tolerance", type=float, default=None)
    parser.add_argument("--ga-compare-stage2-stability-tolerance", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stage1-search-episodes", type=int, default=51000)
    parser.add_argument("--stage2-search-episodes", type=int, default=40000)
    parser.add_argument("--stage1-search-generations", type=int, default=1594)
    parser.add_argument("--stage2-search-generations", type=int, default=1250)
    parser.add_argument("--stage1-search-lr", default="1e-4")
    parser.add_argument("--stage2-search-lr", default="1e-4")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--perm-trials", type=int, default=10)
    parser.add_argument("--cost-trials", type=int, default=10)
    parser.add_argument("--budget-trials", type=int, default=10)
    parser.add_argument("--stage2-compare-repeats", type=int, default=None)
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--rl-cuda-visible-devices", default="")
    parser.add_argument("--ga-cuda-visible-devices", default="")
    parser.add_argument("--rl-skip-stage1-search", action="store_true")
    parser.add_argument("--ga-skip-stage1-search", action="store_true")
    parser.add_argument("--rl-final-eval-source", default="search")
    parser.add_argument("--ga-final-eval-source", default="search")
    parser.add_argument("--rl-final-eval-config", default="")
    parser.add_argument("--ga-final-eval-config", default="")
    parser.add_argument("--rl-skip-noise-search", action="store_true")
    parser.add_argument("--ga-skip-noise-search", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage1-accuracy-tolerance", type=float, default=None)
    parser.add_argument("--stage2-limit-tolerance", type=float, default=None)
    parser.add_argument("--stage2-stability-tolerance", type=float, default=None)
    parser.add_argument("--stage2-k-trials", type=int, default=None,
                        help="Stage-2 稳定性评测噪声试验次数 K（默认 5）")
    parser.add_argument("--stage2-probe-size", type=int, default=None,
                        help="Stage-2 稳定性评测探针子集大小（默认 256）")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.dataset = str(args.dataset or "").strip().lower()
    args.model_type = normalize_model_type(args.model_type)
    if args.stage2_compare_repeats is None:
        args.stage2_compare_repeats = 1
    for flag_name in (
        "batch_size",
        "perm_trials",
        "cost_trials",
        "budget_trials",
        "stage2_compare_repeats",
        "poll_seconds",
    ):
        if getattr(args, flag_name) <= 0:
            raise CompareRunnerError(f"{flag_name} must be a positive integer.")
    for flag_name in ("stage2_k_trials", "stage2_probe_size"):
        v = getattr(args, flag_name, None)
        if v is not None and int(v) <= 0:
            raise CompareRunnerError(f"{flag_name} must be a positive integer when provided.")
    for flag_name in (
        "stage1_accuracy_tolerance",
        "stage2_limit_tolerance",
        "stage2_stability_tolerance",
        "rl_compare_stage1_accuracy_tolerance",
        "rl_compare_stage2_limit_tolerance",
        "rl_compare_stage2_stability_tolerance",
        "ga_compare_stage1_accuracy_tolerance",
        "ga_compare_stage2_limit_tolerance",
        "ga_compare_stage2_stability_tolerance",
    ):
        value = getattr(args, flag_name, None)
        if value is None:
            continue
        if value <= 0 or value >= 1:
            raise CompareRunnerError(
                f"{flag_name} must be a float in (0, 1), got {value!r}."
            )
    if args.compare_config_mode == "persistent":
        direct_values = (
            args.rl_compare_stage1_json,
            args.rl_compare_stage2_json,
            args.ga_compare_stage1_json,
            args.ga_compare_stage2_json,
        )
        if any(str(value or "").strip() for value in direct_values):
            raise CompareRunnerError(
                "persistent compare mode does not accept direct stage JSON path flags."
            )
        if args.rl_compare_stage1_accuracy_tolerance is None:
            args.rl_compare_stage1_accuracy_tolerance = (
                args.stage1_accuracy_tolerance
                if args.stage1_accuracy_tolerance is not None
                else 0.005
            )
        if args.rl_compare_stage2_limit_tolerance is None:
            args.rl_compare_stage2_limit_tolerance = (
                args.stage2_limit_tolerance
                if args.stage2_limit_tolerance is not None
                else 0.05
            )
        if args.rl_compare_stage2_stability_tolerance is None:
            args.rl_compare_stage2_stability_tolerance = (
                args.stage2_stability_tolerance
                if args.stage2_stability_tolerance is not None
                else 0.05
            )
        if args.ga_compare_stage1_accuracy_tolerance is None:
            args.ga_compare_stage1_accuracy_tolerance = (
                args.stage1_accuracy_tolerance
                if args.stage1_accuracy_tolerance is not None
                else 0.005
            )
        if args.ga_compare_stage2_limit_tolerance is None:
            args.ga_compare_stage2_limit_tolerance = (
                args.stage2_limit_tolerance
                if args.stage2_limit_tolerance is not None
                else 0.05
            )
        if args.ga_compare_stage2_stability_tolerance is None:
            args.ga_compare_stage2_stability_tolerance = (
                args.stage2_stability_tolerance
                if args.stage2_stability_tolerance is not None
                else 0.05
            )
        return run_evaluation_only_compare(args)
    if args.compare_config_mode == "direct":
        persistent_values = (
            args.rl_compare_stage1_accuracy_tolerance,
            args.rl_compare_stage2_limit_tolerance,
            args.rl_compare_stage2_stability_tolerance,
            args.ga_compare_stage1_accuracy_tolerance,
            args.ga_compare_stage2_limit_tolerance,
            args.ga_compare_stage2_stability_tolerance,
        )
        if any(value is not None for value in persistent_values):
            raise CompareRunnerError(
                "direct compare mode does not accept persistent constraint flags."
            )
        required_direct_flags = {
            "rl_compare_stage1_json": args.rl_compare_stage1_json,
            "rl_compare_stage2_json": args.rl_compare_stage2_json,
            "ga_compare_stage1_json": args.ga_compare_stage1_json,
            "ga_compare_stage2_json": args.ga_compare_stage2_json,
        }
        missing = [
            name for name, value in required_direct_flags.items()
            if not str(value or "").strip()
        ]
        if missing:
            raise CompareRunnerError(
                f"direct compare mode requires these JSON paths: {', '.join(missing)}."
            )
        return run_evaluation_only_compare(args)
    for flag_name in (
        "stage1_search_episodes",
        "stage2_search_episodes",
        "stage1_search_generations",
        "stage2_search_generations",
    ):
        if getattr(args, flag_name) <= 0:
            raise CompareRunnerError(f"{flag_name} must be a positive integer.")
    compare_root = Path(args.output_dir).resolve()
    meta_dir = compare_root / "meta"
    children_dir = compare_root / "children"
    rl_run_dir = children_dir / "rl"
    ga_run_dir = children_dir / "ga"
    compare_dir = compare_root / "reports"
    compare_metadata_path = meta_dir / "compare_metadata.json"
    compare_runtime_path = meta_dir / "compare_runtime.json"
    compare_status_path = meta_dir / "compare_status.json"
    compare_final_status_path = meta_dir / "compare_final_status.json"
    compare_pid_path = meta_dir / "compare.pid"
    rl_pid_path = meta_dir / "rl.pid"
    ga_pid_path = meta_dir / "ga.pid"
    clear_error_summary(str(compare_root))

    global_stop_requested = {"value": False}
    rl_side_config = normalize_compare_side_config(
        label="RL",
        skip_stage1_search=args.rl_skip_stage1_search,
        skip_noise_search=args.rl_skip_noise_search,
        final_eval_config_source=args.rl_final_eval_source,
        final_eval_config_path=args.rl_final_eval_config,
    )
    ga_side_config = normalize_compare_side_config(
        label="GA",
        skip_stage1_search=args.ga_skip_stage1_search,
        skip_noise_search=args.ga_skip_noise_search,
        final_eval_config_source=args.ga_final_eval_source,
        final_eval_config_path=args.ga_final_eval_config,
    )

    rl_spec = ChildRunSpec(
        algorithm="rl",
        entrypoint="rl_tune.py",
        run_dir=rl_run_dir,
        log_path=rl_run_dir / "logs" / "output.log",
        command=build_child_command(
            python_exe=args.python_exe,
            algorithm="rl",
            side_config=rl_side_config,
            base_model=args.base_model,
            data_path=args.data_path,
            run_dir=rl_run_dir,
            batch_size=args.batch_size,
            stage1_search_episodes=args.stage1_search_episodes,
            stage2_search_episodes=args.stage2_search_episodes,
            stage1_search_generations=args.stage1_search_generations,
            stage2_search_generations=args.stage2_search_generations,
            stage1_search_lr=args.stage1_search_lr,
            stage2_search_lr=args.stage2_search_lr,
            random_seed=args.random_seed,
            perm_trials=args.perm_trials,
            cost_trials=args.cost_trials,
            budget_trials=args.budget_trials,
            stage2_compare_repeats=args.stage2_compare_repeats,
            stage1_accuracy_tolerance=getattr(args, "stage1_accuracy_tolerance", None),
            stage2_limit_tolerance=getattr(args, "stage2_limit_tolerance", None),
            stage2_stability_tolerance=getattr(args, "stage2_stability_tolerance", None),
            stage2_k_trials=getattr(args, "stage2_k_trials", None),
            stage2_probe_size=getattr(args, "stage2_probe_size", None),
        ),
        env_overrides={},
    )
    ga_spec = ChildRunSpec(
        algorithm="ga",
        entrypoint="rl_tune_genetic.py",
        run_dir=ga_run_dir,
        log_path=ga_run_dir / "logs" / "output.log",
        command=build_child_command(
            python_exe=args.python_exe,
            algorithm="ga",
            side_config=ga_side_config,
            base_model=args.base_model,
            data_path=args.data_path,
            run_dir=ga_run_dir,
            batch_size=args.batch_size,
            stage1_search_episodes=args.stage1_search_episodes,
            stage2_search_episodes=args.stage2_search_episodes,
            stage1_search_generations=args.stage1_search_generations,
            stage2_search_generations=args.stage2_search_generations,
            stage1_search_lr=args.stage1_search_lr,
            stage2_search_lr=args.stage2_search_lr,
            random_seed=args.random_seed,
            perm_trials=args.perm_trials,
            cost_trials=args.cost_trials,
            budget_trials=args.budget_trials,
            stage2_compare_repeats=args.stage2_compare_repeats,
            stage1_accuracy_tolerance=getattr(args, "stage1_accuracy_tolerance", None),
            stage2_limit_tolerance=getattr(args, "stage2_limit_tolerance", None),
            stage2_stability_tolerance=getattr(args, "stage2_stability_tolerance", None),
            stage2_k_trials=getattr(args, "stage2_k_trials", None),
            stage2_probe_size=getattr(args, "stage2_probe_size", None),
        ),
        env_overrides={},
    )

    def _handle_sigint(signum, frame):
        del signum, frame
        if global_stop_requested["value"]:
            log("已第二次收到中断信号，正在尝试尽快退出。")
            return
        global_stop_requested["value"] = True
        log("收到中断信号，正在向 RL/GA 子进程转发 SIGINT，请等待它们写 checkpoint 并退出。")
        send_interrupt(rl_spec)
        send_interrupt(ga_spec)

    signal.signal(signal.SIGINT, _handle_sigint)

    compare_warnings: List[str] = []
    rl_cuda = normalize_cuda_value(args.rl_cuda_visible_devices)
    ga_cuda = normalize_cuda_value(args.ga_cuda_visible_devices)
    if not rl_cuda and not ga_cuda:
        rl_cuda, ga_cuda, auto_cuda_warnings = split_cuda_visible_devices(
            os.environ.get("CUDA_VISIBLE_DEVICES")
        )
        compare_warnings.extend(auto_cuda_warnings)
    if rl_cuda:
        rl_spec.env_overrides["CUDA_VISIBLE_DEVICES"] = rl_cuda
    if ga_cuda:
        ga_spec.env_overrides["CUDA_VISIBLE_DEVICES"] = ga_cuda

    metadata = {
        "dataset": args.dataset,
        "base_model": args.base_model,
        "data_path": args.data_path,
        "compare_root": str(compare_root),
        "rl_run_dir": str(rl_run_dir),
        "ga_run_dir": str(ga_run_dir),
        "rl_cuda_visible_devices": rl_cuda,
        "ga_cuda_visible_devices": ga_cuda,
        "rl_stage1_search_episodes": args.stage1_search_episodes,
        "rl_stage2_search_episodes": args.stage2_search_episodes,
        "ga_stage1_search_generations": args.stage1_search_generations,
        "ga_stage2_search_generations": args.stage2_search_generations,
        "rl_side_config": to_jsonable(rl_side_config.__dict__),
        "ga_side_config": to_jsonable(ga_side_config.__dict__),
        "stage1_search_lr": args.stage1_search_lr,
        "stage2_search_lr": args.stage2_search_lr,
        "random_seed": args.random_seed,
        "perm_trials": args.perm_trials,
        "cost_trials": args.cost_trials,
        "budget_trials": args.budget_trials,
        "stage2_compare_repeats": args.stage2_compare_repeats,
        "warnings": compare_warnings,
        "rl_command": rl_spec.command,
        "ga_command": ga_spec.command,
    }
    write_json(compare_metadata_path, metadata)

    if args.dry_run:
        log("dry-run 模式：仅写入 meta/compare_metadata.json，不启动任何子进程。")
        return 0

    log("启动 RL 与 GA 并行对比实验。")
    start_child(rl_spec, extra_env={})
    start_child(ga_spec, extra_env={})
    write_json(
        compare_runtime_path,
        {
            "compare_pid": os.getpid(),
            "rl_pid": rl_spec.process.pid if rl_spec.process else None,
            "ga_pid": ga_spec.process.pid if ga_spec.process else None,
        },
    )
    compare_pid_path.parent.mkdir(parents=True, exist_ok=True)
    compare_pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    rl_pid_path.write_text(f"{rl_spec.process.pid}\n", encoding="utf-8")
    ga_pid_path.write_text(f"{ga_spec.process.pid}\n", encoding="utf-8")

    stage1_compared = False
    stage1_report_path = None
    stage2_report_path = None
    loop_started = time.time()

    try:
        while True:
            rl_state = summarize_process_state(
                rl_spec, args.dataset, global_stop_requested["value"]
            )
            ga_state = summarize_process_state(
                ga_spec, args.dataset, global_stop_requested["value"]
            )
            write_json(
                compare_status_path,
                {
                    "updated_at": now_ts(),
                    "elapsed_seconds": round(time.time() - loop_started, 2),
                    "global_stop_requested": global_stop_requested["value"],
                    "rl": rl_state,
                    "ga": ga_state,
                    "stage1_comparison_generated": bool(stage1_report_path),
                    "stage2_comparison_generated": bool(stage2_report_path),
                },
            )

            if not stage1_compared:
                rl_stage1_ready = final_eval_json(rl_run_dir, args.dataset).is_file() or child_return_code(rl_spec) is not None
                ga_stage1_ready = final_eval_json(ga_run_dir, args.dataset).is_file() or child_return_code(ga_spec) is not None
                if rl_stage1_ready and ga_stage1_ready:
                    rl_json, rl_warn = ensure_final_eval_json(
                        algorithm="rl",
                        run_dir=rl_run_dir,
                        side_config=rl_side_config,
                        dataset=args.dataset,
                        base_model=args.base_model,
                        data_path=args.data_path,
                        batch_size=args.batch_size,
                        stage1_rl_lr=args.stage1_search_lr,
                        stage2_rl_lr=args.stage2_search_lr,
                        random_seed=args.random_seed,
                        perm_trials=args.perm_trials,
                        cost_trials=args.cost_trials,
                        budget_trials=args.budget_trials,
                        final_eval_repeat_n=args.stage2_compare_repeats,
                    )
                    ga_json, ga_warn = ensure_final_eval_json(
                        algorithm="ga",
                        run_dir=ga_run_dir,
                        side_config=ga_side_config,
                        dataset=args.dataset,
                        base_model=args.base_model,
                        data_path=args.data_path,
                        batch_size=args.batch_size,
                        stage1_rl_lr=args.stage1_search_lr,
                        stage2_rl_lr=args.stage2_search_lr,
                        random_seed=args.random_seed,
                        perm_trials=args.perm_trials,
                        cost_trials=args.cost_trials,
                        budget_trials=args.budget_trials,
                        final_eval_repeat_n=args.stage2_compare_repeats,
                    )
                    payload = build_stage_compare_payload(
                        stage_label="stage1",
                        dataset=args.dataset,
                        compare_root=compare_root,
                        rl_run_dir=rl_run_dir,
                        ga_run_dir=ga_run_dir,
                        rl_json_path=rl_json,
                        ga_json_path=ga_json,
                        rl_warnings=rl_warn,
                        ga_warnings=ga_warn,
                        process_meta={
                            "compare_warnings": compare_warnings,
                            "rl": rl_state,
                            "ga": ga_state,
                        },
                    )
                    stage1_report_path, _ = save_stage_compare_report(payload, compare_dir)
                    log(f"Stage-1 对比结果已生成：{stage1_report_path}")
                    stage1_compared = True

            if child_return_code(rl_spec) is not None and child_return_code(ga_spec) is not None:
                break

            time.sleep(max(1, int(args.poll_seconds)))

        rl_state = summarize_process_state(
            rl_spec, args.dataset, global_stop_requested["value"]
        )
        ga_state = summarize_process_state(
            ga_spec, args.dataset, global_stop_requested["value"]
        )

        if not stage1_compared:
            rl_json, rl_warn = ensure_final_eval_json(
                algorithm="rl",
                run_dir=rl_run_dir,
                side_config=rl_side_config,
                dataset=args.dataset,
                base_model=args.base_model,
                data_path=args.data_path,
                batch_size=args.batch_size,
                stage1_rl_lr=args.stage1_search_lr,
                stage2_rl_lr=args.stage2_search_lr,
                random_seed=args.random_seed,
                perm_trials=args.perm_trials,
                cost_trials=args.cost_trials,
                budget_trials=args.budget_trials,
                final_eval_repeat_n=args.stage2_compare_repeats,
            )
            ga_json, ga_warn = ensure_final_eval_json(
                algorithm="ga",
                run_dir=ga_run_dir,
                side_config=ga_side_config,
                dataset=args.dataset,
                base_model=args.base_model,
                data_path=args.data_path,
                batch_size=args.batch_size,
                stage1_rl_lr=args.stage1_search_lr,
                stage2_rl_lr=args.stage2_search_lr,
                random_seed=args.random_seed,
                perm_trials=args.perm_trials,
                cost_trials=args.cost_trials,
                budget_trials=args.budget_trials,
                final_eval_repeat_n=args.stage2_compare_repeats,
            )
            payload = build_stage_compare_payload(
                stage_label="stage1",
                dataset=args.dataset,
                compare_root=compare_root,
                rl_run_dir=rl_run_dir,
                ga_run_dir=ga_run_dir,
                rl_json_path=rl_json,
                ga_json_path=ga_json,
                rl_warnings=rl_warn,
                ga_warnings=ga_warn,
                process_meta={
                    "compare_warnings": compare_warnings,
                    "rl": rl_state,
                    "ga": ga_state,
                },
            )
            stage1_report_path, _ = save_stage_compare_report(payload, compare_dir)
            log(f"Stage-1 对比结果已补生成：{stage1_report_path}")

        rl_json, rl_warn = ensure_final_eval_json(
            algorithm="rl",
            run_dir=rl_run_dir,
            side_config=rl_side_config,
            dataset=args.dataset,
            base_model=args.base_model,
            data_path=args.data_path,
            batch_size=args.batch_size,
            stage1_rl_lr=args.stage1_search_lr,
            stage2_rl_lr=args.stage2_search_lr,
            random_seed=args.random_seed,
            perm_trials=args.perm_trials,
            cost_trials=args.cost_trials,
            budget_trials=args.budget_trials,
            final_eval_repeat_n=args.stage2_compare_repeats,
        )
        ga_json, ga_warn = ensure_final_eval_json(
            algorithm="ga",
            run_dir=ga_run_dir,
            side_config=ga_side_config,
            dataset=args.dataset,
            base_model=args.base_model,
            data_path=args.data_path,
            batch_size=args.batch_size,
            stage1_rl_lr=args.stage1_search_lr,
            stage2_rl_lr=args.stage2_search_lr,
            random_seed=args.random_seed,
            perm_trials=args.perm_trials,
            cost_trials=args.cost_trials,
            budget_trials=args.budget_trials,
            final_eval_repeat_n=args.stage2_compare_repeats,
        )
        payload = build_stage_compare_payload(
            stage_label="stage2",
            dataset=args.dataset,
            compare_root=compare_root,
            rl_run_dir=rl_run_dir,
            ga_run_dir=ga_run_dir,
            rl_json_path=rl_json,
            ga_json_path=ga_json,
            rl_warnings=rl_warn,
            ga_warnings=ga_warn,
            process_meta={
                "compare_warnings": compare_warnings,
                "rl": rl_state,
                "ga": ga_state,
            },
        )
        stage2_report_path, _ = save_stage_compare_report(payload, compare_dir)
        log(f"Stage-2 对比结果已生成：{stage2_report_path}")

        final_state = {
            "updated_at": now_ts(),
            "global_stop_requested": global_stop_requested["value"],
            "rl": rl_state,
            "ga": ga_state,
            "final_report_path": str(stage2_report_path) if stage2_report_path else None,
            "stage1_report_path": str(stage1_report_path) if stage1_report_path else None,
            "stage2_report_path": str(stage2_report_path) if stage2_report_path else None,
        }
        write_json(compare_final_status_path, final_state)
        abnormal_children = [
            state for state in (rl_state, ga_state)
            if is_abnormal_final_state(state)
        ]
        if global_stop_requested["value"] or abnormal_children:
            if global_stop_requested["value"]:
                status = "interrupted"
                message = (
                    "Received SIGINT and forwarded it to the RL/GA child processes. "
                    "The compare run stopped before full completion."
                )
                signal_name = "SIGINT"
                exit_code = 130
            else:
                status = "failed"
                message = (
                    "At least one RL/GA child run exited abnormally or stopped early. "
                    "See the child summaries below."
                )
                signal_name = ""
                exit_code = 1
            write_error_summary(
                str(compare_root),
                program_name="rl_ga_compare_runner.py",
                status=status,
                message=message,
                argv=sys.argv,
                exit_code=exit_code,
                signal_name=signal_name,
                extra_sections=build_compare_error_sections(
                    rl_state, ga_state, rl_spec, ga_spec
                ),
            )
        log("RL/GA 对比实验已结束。")
        if global_stop_requested["value"]:
            return 130
        if abnormal_children:
            return 1
        return 0
    except Exception as exc:
        error_payload = {
            "updated_at": now_ts(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(meta_dir / "compare_error.json", error_payload)
        write_error_summary(
            str(compare_root),
            program_name="rl_ga_compare_runner.py",
            status="failed",
            message=f"{type(exc).__name__}: {exc}",
            argv=sys.argv,
            exit_code=1,
            traceback_text=traceback.format_exc(),
            extra_sections=build_compare_error_sections(
                None, None, rl_spec, ga_spec
            ),
        )
        log(f"[Error] 对比实验失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
