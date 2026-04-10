import argparse
import json
import os
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


TARGET_MODULES_LITERAL = "[\"q_proj\", \"k_proj\", \"v_proj\", \"up_proj\", \"down_proj\"]"
STAGE1_RL_CHECKPOINT_FILENAME = "stage1_rl_checkpoint.pt"
NOISE_STAGE_CHECKPOINT_FILENAME = "noise_rl_checkpoint.pt"
GA_STAGE1_CHECKPOINT_FILENAME = "ga_stage1_checkpoint.pt"
GA_STAGE2_CHECKPOINT_FILENAME = "ga_stage2_checkpoint.pt"
DEFAULT_POLL_SECONDS = 15

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


class CompareRunnerError(RuntimeError):
    pass


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now_ts()}] {msg}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def stage1_final_eval_json(run_dir: Path, dataset: str) -> Path:
    return run_dir / "stage1_final_eval" / f"final_eval_results_{dataset}.json"


def stage2_final_eval_json(run_dir: Path, dataset: str) -> Path:
    return run_dir / "stage2_noise_final_eval" / f"noise_final_eval_results_{dataset}.json"


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
    obj = read_json(stage1_final_eval_json(run_dir, dataset))
    if obj is not None and obj.get("selected") is not None:
        return {
            "gelu": obj["selected"]["gelu"],
            "softmax": obj["selected"]["softmax"],
        }, "stage1_final_eval_selected"
    cfg, source = recover_stage1_search_best(run_dir, algorithm)
    return cfg, source


def build_compare_evaluator(
    *,
    base_model: str,
    data_path: str,
    batch_size: int,
    run_output_dir: str,
    search_algorithm: str,
    stage1_rl_lr: Optional[str],
    stage2_rl_lr: Optional[str],
    random_seed: int,
    perm_trials: int,
    cost_trials: int,
    budget_trials: int,
    noise_eval_repeat_n: int,
):
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
        train_data = data["train"].shuffle().map(tokenize_fn)
        val_data = data["validation_matched"].shuffle().map(tokenize_fn)
        val_data_mm = data["validation_mismatched"].shuffle().map(tokenize_fn)
        train_data = train_data.rename_column("label", "labels")
        val_data = val_data.rename_column("label", "labels")
        val_data_mm = val_data_mm.rename_column("label", "labels")
        mm_columns = [c for c in ("input_ids", "attention_mask", "token_type_ids", "labels") if c in val_data_mm.column_names]
        val_data_mm.set_format(type="torch", columns=mm_columns)
    else:
        train_data = data["train"].shuffle().map(tokenize_fn)
        val_data = data["validation"].shuffle().map(tokenize_fn)
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
        final_eval_random_seed=random_seed,
        final_eval_permutation_trials=perm_trials,
        final_eval_cost_equivalent_trials=cost_trials,
        final_eval_budget_equivalent_trials=budget_trials,
        noise_eval_repeat_n=noise_eval_repeat_n,
        skip_stage1_rl=True,
        skip_noise_rl=True,
        skip_stage1_final_eval=False,
        skip_noise_final_eval=False,
        data_path=data_path,
        search_algorithm=search_algorithm,
    )
    return evaluator


def ensure_stage1_eval_json(
    *,
    algorithm: str,
    run_dir: Path,
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
    noise_eval_repeat_n: int,
) -> Tuple[Path, List[str]]:
    json_path = stage1_final_eval_json(run_dir, dataset)
    if json_path.is_file():
        return json_path, []

    warnings: List[str] = [
        f"{algorithm.upper()} 的 Stage-1 最终评估文件缺失，已改为基于当前最优配置补做最终评估。"
    ]
    search_best_config, source = recover_stage1_search_best(run_dir, algorithm)

    evaluator = build_compare_evaluator(
        base_model=base_model,
        data_path=data_path,
        batch_size=batch_size,
        run_output_dir=str(run_dir),
        search_algorithm=algorithm,
        stage1_rl_lr=stage1_rl_lr,
        stage2_rl_lr=stage2_rl_lr,
        random_seed=random_seed,
        perm_trials=perm_trials,
        cost_trials=cost_trials,
        budget_trials=budget_trials,
        noise_eval_repeat_n=noise_eval_repeat_n,
    )

    from final_evaluation_module import FinalEvaluationModule
    from genetic_search_module import GeneticFinalEvaluationModule, build_stage1_context

    context = build_stage1_context(evaluator, log_fn=evaluator.log, include_distribution=False)
    if search_best_config is None:
        warnings.append(
            f"{algorithm.upper()} 未找到 Stage-1 checkpoint/search 结果，已回退到 baseline 配置生成对比结果。"
        )
        search_best_config = {
            "gelu": context.base_gelu.copy(),
            "softmax": context.base_softmax.copy(),
            "cost": float(context.base_tot_c),
        }
    else:
        warnings.append(f"{algorithm.upper()} Stage-1 fallback 来源：{source}")

    module_cls = GeneticFinalEvaluationModule if algorithm == "ga" else FinalEvaluationModule
    runner = module_cls(
        evaluator=evaluator,
        config_source="search",
        random_seed=random_seed,
        permutation_trials=perm_trials,
        cost_equivalent_trials=cost_trials,
        budget_equivalent_trials=budget_trials,
        results_dir=evaluator.stage1_final_eval_dir,
    )
    result = runner.run(
        search_best_config=search_best_config,
        base_gelu=context.base_gelu,
        base_softmax=context.base_softmax,
        base_tot_c=context.base_tot_c,
        base_g_c=context.base_g_c,
        base_s_c=context.base_s_c,
        limit_loss=context.limit_loss,
        limit_p=context.limit_p,
        limit_s=context.limit_s,
    )
    return Path(result["summary_path"]), warnings


def ensure_stage2_eval_json(
    *,
    algorithm: str,
    run_dir: Path,
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
    noise_eval_repeat_n: int,
) -> Tuple[Path, List[str]]:
    json_path = stage2_final_eval_json(run_dir, dataset)
    if json_path.is_file():
        return json_path, []

    warnings: List[str] = [
        f"{algorithm.upper()} 的 Stage-2 最终评估文件缺失，已改为基于当前最优噪声配置补做最终评估。"
    ]

    fixed_stage1_config, fixed_source = resolve_stage1_selected_config_from_artifacts(
        run_dir, algorithm, dataset
    )

    evaluator = build_compare_evaluator(
        base_model=base_model,
        data_path=data_path,
        batch_size=batch_size,
        run_output_dir=str(run_dir),
        search_algorithm=algorithm,
        stage1_rl_lr=stage1_rl_lr,
        stage2_rl_lr=stage2_rl_lr,
        random_seed=random_seed,
        perm_trials=perm_trials,
        cost_trials=cost_trials,
        budget_trials=budget_trials,
        noise_eval_repeat_n=noise_eval_repeat_n,
    )

    from genetic_search_module import (
        GeneticNoiseFinalEvaluationModule,
        build_stage1_context,
        build_stage2_context,
    )
    from noise_final_evaluation_module import NoiseFinalEvaluationModule

    if fixed_stage1_config is None:
        stage1_context = build_stage1_context(
            evaluator, log_fn=evaluator.log, include_distribution=False
        )
        fixed_stage1_config = {
            "gelu": stage1_context.base_gelu.copy(),
            "softmax": stage1_context.base_softmax.copy(),
        }
        warnings.append(
            f"{algorithm.upper()} 未找到 Stage-1 选中配置，Stage-2 fallback 改用 baseline Stage-1 配置。"
        )
    else:
        warnings.append(f"{algorithm.upper()} Stage-2 固定的 Stage-1 配置来源：{fixed_source}")

    fixed_gelu = np.asarray(fixed_stage1_config["gelu"], dtype=int)
    fixed_softmax = np.asarray(fixed_stage1_config["softmax"], dtype=int)
    noise_best_config, noise_source = recover_stage2_search_best(run_dir, algorithm)
    if noise_best_config is None:
        warnings.append(
            f"{algorithm.upper()} 未找到 Stage-2 稳定最优噪声配置，将仅输出 baseline/no-noise 对照并附带警告。"
        )
    else:
        warnings.append(f"{algorithm.upper()} Stage-2 fallback 来源：{noise_source}")

    context = build_stage2_context(
        evaluator,
        fixed_gelu,
        fixed_softmax,
        log_fn=evaluator.log,
    )
    module_cls = GeneticNoiseFinalEvaluationModule if algorithm == "ga" else NoiseFinalEvaluationModule
    runner = module_cls(
        evaluator=evaluator,
        config_source="search",
        random_seed=random_seed,
        permutation_trials=perm_trials,
        cost_equivalent_trials=cost_trials,
        budget_equivalent_trials=budget_trials,
        repeat_n=noise_eval_repeat_n,
        results_dir=evaluator.noise_final_eval_dir,
    )
    result = runner.run(
        search_best_noise_config=noise_best_config,
        search_status="fallback_missing_search_best" if noise_best_config is None else "fallback_from_partial_run",
        fixed_gelu=fixed_gelu,
        fixed_softmax=fixed_softmax,
        baseline_noise_config=context.cost_reference_noise_config,
        baseline_tot_c=context.cost_reference_tot_c,
        limit_loss=context.search_limits["loss"],
        limit_p=context.search_limits["metric1"],
        limit_s=context.search_limits["metric2"],
    )
    return Path(result["summary_path"]), warnings


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
        stage1_selected_config, stage1_selected_source = (
            resolve_stage1_selected_config_from_artifacts(run_dir, algorithm, dataset)
        )
        return {
            "label": label,
            "baseline": baseline,
            "selected": selected,
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
    return payload


def save_stage_compare_report(payload: dict, output_dir: Path) -> Tuple[Path, Path]:
    ensure_dir(output_dir)
    stage = payload["stage"]
    dataset = payload["dataset"]
    metric_names = payload["metric_names"]
    rl_side = payload["sides"]["rl"]
    ga_side = payload["sides"]["ga"]
    json_path = output_dir / f"{stage}_compare_summary_{dataset}.json"
    md_path = output_dir / f"{stage}_compare_report_{dataset}.md"
    plot_path = output_dir / f"{stage}_compare_plot_{dataset}.png"

    write_json(json_path, payload)

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
            f"{float(selected.get('loss', 0.0)):.6f}",
            f"{float(selected.get('p', 0.0)):.6f}",
            metric2_text,
            f"{float(selected.get('tot_c', 0.0)):.4f}" if selected.get("tot_c") is not None else "-",
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
        "Loss",
        metric_names[0],
        header_metric2,
        "Cost",
        "Time(ms)",
        "Feasible",
        "dLoss%",
        f"d{metric_names[0]}%",
        f"d{header_metric2}%" if len(metric_names) > 1 else "-",
    ]
    rows = [row("RL", rl_side), row("GA", ga_side)]

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
        rl_breakdown = (rl_side["selected"] or {}).get("breakdown")
        ga_breakdown = (ga_side["selected"] or {}).get("breakdown")
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

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _plot_stage_compare(payload, plot_path)
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

    metrics = [
        ("loss", "Loss"),
        ("p", metric_names[0]),
    ]
    if len(metric_names) > 1:
        metrics.append(("s", metric_names[1]))
    else:
        metrics.append(("time_ms", "Time(ms)"))
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
        for idx, value in enumerate([rl_value, ga_value]):
            ax.text(idx, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close(fig)


def build_child_command(
    *,
    python_exe: str,
    algorithm: str,
    base_model: str,
    data_path: str,
    run_dir: Path,
    batch_size: int,
    stage1_search_episodes: int,
    stage2_search_episodes: int,
    stage1_search_lr: Optional[str],
    stage2_search_lr: Optional[str],
    random_seed: int,
    perm_trials: int,
    cost_trials: int,
    budget_trials: int,
    noise_eval_repeat: int,
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
        "--stage1_rl_episodes", str(stage1_search_episodes),
        "--stage2_rl_episodes", str(stage2_search_episodes),
        "--stage1_rl_episodes_specified", "true",
        "--stage2_rl_episodes_specified", "true",
        "--use_ist",
        "--final_eval_config_source", "search",
        "--final_eval_config_path", "",
        "--manual_final_gelu", "",
        "--manual_final_softmax", "",
        "--final_eval_random_seed", str(random_seed),
        "--final_eval_permutation_trials", str(perm_trials),
        "--final_eval_cost_equivalent_trials", str(cost_trials),
        "--final_eval_budget_equivalent_trials", str(budget_trials),
        "--skip_noise_rl", "false",
        "--noise_eval_config_source", "search",
        "--noise_eval_config_path", "",
        "--manual_noise_config", "",
        "--noise_eval_repeat_n", str(noise_eval_repeat),
        "--skip_stage1_rl", "false",
        "--skip_stage1_final_eval", "false",
        "--skip_noise_final_eval", "false",
        "--resume_run_dir", "",
    ]
    if algorithm == "rl":
        cmd.extend(["--stage1_rl_lr", str(stage1_search_lr), "--stage2_rl_lr", str(stage2_search_lr)])
    return cmd


def start_child(spec: ChildRunSpec, extra_env: Dict[str, str]) -> None:
    ensure_dir(spec.run_dir / "logs")
    ensure_dir(spec.log_path.parent)
    env = os.environ.copy()
    env.update(extra_env)
    env.update(spec.env_overrides)
    with spec.log_path.open("w", encoding="utf-8") as log_handle:
        spec.process = subprocess.Popen(
            spec.command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
            env=env,
            start_new_session=True,
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
    s1_ok = stage1_final_eval_json(spec.run_dir, dataset).is_file()
    s2_ok = stage2_final_eval_json(spec.run_dir, dataset).is_file()
    if rc is None:
        state = "running"
    elif rc == 0 and s1_ok and s2_ok:
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
        "stage1_final_eval_ready": s1_ok,
        "stage2_final_eval_ready": s2_ok,
        "run_dir": str(spec.run_dir),
        "log_path": str(spec.log_path),
        "command": spec.command,
        "env_overrides": spec.env_overrides,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="并行运行 RL 与 GA，并生成阶段对比结果。")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stage1-search-episodes", type=int, required=True)
    parser.add_argument("--stage2-search-episodes", type=int, required=True)
    parser.add_argument("--stage1-search-lr", default="1e-4")
    parser.add_argument("--stage2-search-lr", default="1e-4")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--perm-trials", type=int, default=10)
    parser.add_argument("--cost-trials", type=int, default=10)
    parser.add_argument("--budget-trials", type=int, default=10)
    parser.add_argument("--noise-eval-repeat", type=int, default=1)
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--rl-cuda-visible-devices", default="")
    parser.add_argument("--ga-cuda-visible-devices", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compare_root = Path(args.output_dir).resolve()
    rl_run_dir = compare_root / "rl_run"
    ga_run_dir = compare_root / "ga_run"
    compare_dir = compare_root / "comparison"
    logs_dir = compare_root / "logs"
    ensure_dir(compare_dir)
    ensure_dir(logs_dir)

    global_stop_requested = {"value": False}

    rl_spec = ChildRunSpec(
        algorithm="rl",
        entrypoint="rl_tune.py",
        run_dir=rl_run_dir,
        log_path=rl_run_dir / "logs" / "output.log",
        command=build_child_command(
            python_exe=args.python_exe,
            algorithm="rl",
            base_model=args.base_model,
            data_path=args.data_path,
            run_dir=rl_run_dir,
            batch_size=args.batch_size,
            stage1_search_episodes=args.stage1_search_episodes,
            stage2_search_episodes=args.stage2_search_episodes,
            stage1_search_lr=args.stage1_search_lr,
            stage2_search_lr=args.stage2_search_lr,
            random_seed=args.random_seed,
            perm_trials=args.perm_trials,
            cost_trials=args.cost_trials,
            budget_trials=args.budget_trials,
            noise_eval_repeat=args.noise_eval_repeat,
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
            base_model=args.base_model,
            data_path=args.data_path,
            run_dir=ga_run_dir,
            batch_size=args.batch_size,
            stage1_search_episodes=args.stage1_search_episodes,
            stage2_search_episodes=args.stage2_search_episodes,
            stage1_search_lr=args.stage1_search_lr,
            stage2_search_lr=args.stage2_search_lr,
            random_seed=args.random_seed,
            perm_trials=args.perm_trials,
            cost_trials=args.cost_trials,
            budget_trials=args.budget_trials,
            noise_eval_repeat=args.noise_eval_repeat,
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
        "stage1_search_episodes": args.stage1_search_episodes,
        "stage2_search_episodes": args.stage2_search_episodes,
        "stage1_search_lr": args.stage1_search_lr,
        "stage2_search_lr": args.stage2_search_lr,
        "random_seed": args.random_seed,
        "perm_trials": args.perm_trials,
        "cost_trials": args.cost_trials,
        "budget_trials": args.budget_trials,
        "noise_eval_repeat": args.noise_eval_repeat,
        "warnings": compare_warnings,
        "rl_command": rl_spec.command,
        "ga_command": ga_spec.command,
    }
    write_json(compare_root / "compare_metadata.json", metadata)

    if args.dry_run:
        log("dry-run 模式：仅写入 compare_metadata.json，不启动任何子进程。")
        return 0

    log("启动 RL 与 GA 并行对比实验。")
    start_child(rl_spec, extra_env={})
    start_child(ga_spec, extra_env={})
    write_json(
        compare_root / "compare_runtime.json",
        {
            "compare_pid": os.getpid(),
            "rl_pid": rl_spec.process.pid if rl_spec.process else None,
            "ga_pid": ga_spec.process.pid if ga_spec.process else None,
        },
    )
    (compare_root / "compare.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    (compare_root / "rl.pid").write_text(f"{rl_spec.process.pid}\n", encoding="utf-8")
    (compare_root / "ga.pid").write_text(f"{ga_spec.process.pid}\n", encoding="utf-8")

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
                compare_root / "compare_status.json",
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
                rl_stage1_ready = stage1_final_eval_json(rl_run_dir, args.dataset).is_file() or child_return_code(rl_spec) is not None
                ga_stage1_ready = stage1_final_eval_json(ga_run_dir, args.dataset).is_file() or child_return_code(ga_spec) is not None
                if rl_stage1_ready and ga_stage1_ready:
                    rl_json, rl_warn = ensure_stage1_eval_json(
                        algorithm="rl",
                        run_dir=rl_run_dir,
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
                        noise_eval_repeat_n=args.noise_eval_repeat,
                    )
                    ga_json, ga_warn = ensure_stage1_eval_json(
                        algorithm="ga",
                        run_dir=ga_run_dir,
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
                        noise_eval_repeat_n=args.noise_eval_repeat,
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
            rl_json, rl_warn = ensure_stage1_eval_json(
                algorithm="rl",
                run_dir=rl_run_dir,
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
                noise_eval_repeat_n=args.noise_eval_repeat,
            )
            ga_json, ga_warn = ensure_stage1_eval_json(
                algorithm="ga",
                run_dir=ga_run_dir,
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
                noise_eval_repeat_n=args.noise_eval_repeat,
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

        rl_json, rl_warn = ensure_stage2_eval_json(
            algorithm="rl",
            run_dir=rl_run_dir,
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
            noise_eval_repeat_n=args.noise_eval_repeat,
        )
        ga_json, ga_warn = ensure_stage2_eval_json(
            algorithm="ga",
            run_dir=ga_run_dir,
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
            noise_eval_repeat_n=args.noise_eval_repeat,
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
            "stage1_report_path": str(stage1_report_path) if stage1_report_path else None,
            "stage2_report_path": str(stage2_report_path) if stage2_report_path else None,
        }
        write_json(compare_root / "compare_final_status.json", final_state)
        log("RL/GA 对比实验已结束。")
        return 0
    except Exception as exc:
        error_payload = {
            "updated_at": now_ts(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(compare_root / "compare_error.json", error_payload)
        log(f"[Error] 对比实验失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
