import os
import sys
import json
import re
import glob
from datetime import datetime, timezone
from typing import List

import fire
import torch
import transformers
from datasets import DownloadConfig, load_dataset, load_from_disk
from typing import List, Optional, Union
from runtime_error_reporter import run_fire_entrypoint
"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
sys.path.append(os.path.join(os.getcwd(), "./importance-aware-sparse-tuning-IST-paper/peft/src/"))
# from peft import (  # noqa: E402
#     LoraConfig,
#     DoraConfig,
#     BottleneckConfig,
#     PrefixTuningConfig,
#     get_peft_model,
#     get_peft_model_state_dict,
#     prepare_model_for_int8_training,
#     set_peft_model_state_dict,
# )
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, LlamaTokenizer, DataCollatorWithPadding, AutoModel  # noqa: F402


ENABLE_GLUE_EQUIVALENT_PARQUET_ROUTE = True
GLUE_EQUIVALENT_PARQUET_ENDPOINTS = [
    "https://huggingface.co",
]
GLUE_LOCAL_DATASET_ENV_VARS = (
    "GLUE_LOCAL_DATASET_DIR",
    "GLUE_DATASET_DIR",
)


def seed_everything(seed: int) -> int:
    seed = int(seed)
    transformers.set_seed(seed)
    try:
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    return seed



def parse_degree_config(raw_value):
    if raw_value is None or raw_value == "":
        return None
    if isinstance(raw_value, (list, tuple)):
        return [int(item) for item in raw_value]

    text = str(raw_value).strip()
    if not text:
        return None
    if text.startswith("["):
        return [int(item) for item in json.loads(text)]
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_noise_config(raw_value):
    if raw_value is None or raw_value == "":
        return None
    if isinstance(raw_value, dict):
        return raw_value
    text = str(raw_value).strip()
    if not text:
        return None
    return json.loads(text)


GLUE_PARQUET_SPLITS = {
    "cola": ("train", "validation", "test"),
    "sst2": ("train", "validation", "test"),
    "mrpc": ("train", "validation", "test"),
    "stsb": ("train", "validation", "test"),
    "qqp": ("train", "validation", "test"),
    "qnli": ("train", "validation", "test"),
    "rte": ("train", "validation", "test"),
    "wnli": ("train", "validation", "test"),
    "mnli": (
        "train",
        "validation_matched",
        "validation_mismatched",
        "test_matched",
        "test_mismatched",
    ),
}

GLUE_REQUIRED_COLUMNS = {
    "cola": ("sentence", "label"),
    "sst2": ("sentence", "label"),
    "mrpc": ("sentence1", "sentence2", "label"),
    "stsb": ("sentence1", "sentence2", "label"),
    "qqp": ("question1", "question2", "label"),
    "qnli": ("question", "sentence", "label"),
    "rte": ("sentence1", "sentence2", "label"),
    "wnli": ("sentence1", "sentence2", "label"),
    "mnli": ("premise", "hypothesis", "label"),
}


def _hf_endpoint_base() -> str:
    return (
        os.environ.get("HF_ENDPOINT")
        or os.environ.get("HF_HUB_ENDPOINT")
        or "https://huggingface.co"
    ).rstrip("/")


def _extract_hf_endpoint_from_error(exc: Exception) -> str:
    match = re.search(r"(https?://[^/]+)/api/datasets/nyu-mll/glue/", str(exc))
    if match:
        return match.group(1).rstrip("/")
    return _hf_endpoint_base()


def _extract_hf_revision_from_error(exc: Exception) -> str:
    match = re.search(r"/tree/([^/?#]+)/", str(exc))
    if match:
        return match.group(1)
    return "main"


def _glue_equivalent_candidate_endpoints(primary_endpoint: str):
    candidates = []
    for raw in list(GLUE_EQUIVALENT_PARQUET_ENDPOINTS or []):
        endpoint = str(raw or "").strip().rstrip("/")
        if endpoint and endpoint not in candidates:
            candidates.append(endpoint)
    primary = str(primary_endpoint or "").strip().rstrip("/")
    if primary and primary not in candidates:
        candidates.append(primary)
    if not candidates:
        candidates.append(_hf_endpoint_base())
    return candidates


def _glue_parquet_data_files(task_name: str, endpoint: str = None, revision: str = None):
    task = str(task_name).strip().lower()
    splits = GLUE_PARQUET_SPLITS.get(task)
    if not splits:
        return None
    base = (endpoint or _hf_endpoint_base()).rstrip("/")
    rev = str(revision or "main").strip() or "main"
    return {
        split: (
            f"{base}/datasets/nyu-mll/glue/resolve/{rev}/"
            f"{task}/{split}-00000-of-00001.parquet"
        )
        for split in splits
    }


def _validate_glue_dataset_equivalence(data, task_name: str) -> None:
    task = str(task_name).strip().lower()
    expected_splits = GLUE_PARQUET_SPLITS.get(task)
    required_columns = GLUE_REQUIRED_COLUMNS.get(task)
    column_names = getattr(data, "column_names", None)
    if not expected_splits or not required_columns or not isinstance(column_names, dict):
        return

    missing_splits = [split for split in expected_splits if split not in column_names]
    missing_columns = {}
    for split in expected_splits:
        if split not in column_names:
            continue
        cols = set(str(c) for c in column_names.get(split, []))
        missing = [col for col in required_columns if col not in cols]
        if missing:
            missing_columns[split] = missing

    if missing_splits or missing_columns:
        raise ValueError(
            "equivalent parquet schema check failed for "
            f"GLUE task {task!r}: missing_splits={missing_splits}, "
            f"missing_columns={missing_columns}"
        )


def _split_existing_path_list(raw_value: str):
    paths = []
    for raw_item in str(raw_value or "").split(os.pathsep):
        path = raw_item.strip().strip('"').strip("'")
        if path and path not in paths:
            paths.append(path)
    return paths


def _glue_local_dataset_roots():
    roots = []
    for env_name in GLUE_LOCAL_DATASET_ENV_VARS:
        for path in _split_existing_path_list(os.environ.get(env_name, "")):
            if path not in roots:
                roots.append(path)
    return roots


def _glue_local_dataset_candidates(task_name: str):
    task = str(task_name).strip().lower()
    candidates = []
    for root in _glue_local_dataset_roots():
        root = os.path.abspath(os.path.expanduser(root))
        root_name = os.path.basename(os.path.normpath(root)).lower()
        raw_candidates = []
        if root_name == task:
            raw_candidates.append(root)
        raw_candidates.extend([
            os.path.join(root, task),
            os.path.join(root, "glue", task),
            os.path.join(root, "nyu-mll", "glue", task),
        ])
        for path in raw_candidates:
            if path not in candidates:
                candidates.append(path)
    return candidates


def _glue_local_parquet_data_files(task_name: str, dataset_dir: str):
    task = str(task_name).strip().lower()
    splits = GLUE_PARQUET_SPLITS.get(task)
    if not splits:
        return None
    data_files = {}
    for split in splits:
        direct = os.path.join(dataset_dir, f"{split}.parquet")
        if os.path.exists(direct):
            data_files[split] = direct
            continue
        matches = sorted(glob.glob(os.path.join(dataset_dir, f"{split}-*.parquet")))
        if matches:
            data_files[split] = matches if len(matches) > 1 else matches[0]
            continue
        return None
    return data_files


def _write_glue_local_route_log(
        *,
        route_log_dir: str,
        task: str,
        route: str,
        path: str,
        primary_exc: Exception,
        detail: str = "",
        ) -> str:
    os.makedirs(route_log_dir, exist_ok=True)
    log_path = os.path.join(route_log_dir, "glue_dataset_local_route.txt")
    lines = [
        "[glue_dataset_local_route]",
        f"time_utc={datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"task={task}",
        f"route={route}",
        f"path={path}",
        f"detail={detail}",
        "primary_loader=load_dataset('nyu-mll/glue', task)",
        f"primary_error={type(primary_exc).__name__}: {primary_exc}",
    ]
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")
    return log_path


def _try_load_local_glue_dataset(
        task: str,
        *,
        load_dataset_fn,
        load_from_disk_fn,
        route_log_dir: str,
        primary_exc: Exception,
        ):
    errors = []
    for path in _glue_local_dataset_candidates(task):
        if not os.path.isdir(path):
            continue
        try:
            data = load_from_disk_fn(path)
            _validate_glue_dataset_equivalence(data, task)
            log_path = None
            if route_log_dir:
                log_path = _write_glue_local_route_log(
                    route_log_dir=route_log_dir,
                    task=task,
                    route="local_saved_to_disk",
                    path=path,
                    primary_exc=primary_exc,
                )
            print(
                f"[dataset] using local GLUE dataset for task {task!r} from {path}"
                + (f"; audit_log={log_path}" if log_path else ""),
                file=sys.stderr,
            )
            return data, errors
        except Exception as local_exc:
            errors.append(f"{path} load_from_disk: {local_exc!r}")

        data_files = _glue_local_parquet_data_files(task, path)
        if data_files is None:
            continue
        try:
            data = load_dataset_fn("parquet", data_files=data_files)
            _validate_glue_dataset_equivalence(data, task)
            log_path = None
            if route_log_dir:
                log_path = _write_glue_local_route_log(
                    route_log_dir=route_log_dir,
                    task=task,
                    route="local_parquet",
                    path=path,
                    primary_exc=primary_exc,
                    detail=json.dumps(data_files, ensure_ascii=True, sort_keys=True),
                )
            print(
                f"[dataset] using local GLUE parquet files for task {task!r} from {path}"
                + (f"; audit_log={log_path}" if log_path else ""),
                file=sys.stderr,
            )
            return data, errors
        except Exception as local_exc:
            errors.append(f"{path} local_parquet: {local_exc!r}")

    try:
        data = load_dataset_fn(
            "nyu-mll/glue",
            task,
            download_config=DownloadConfig(local_files_only=True),
        )
        _validate_glue_dataset_equivalence(data, task)
        log_path = None
        if route_log_dir:
            log_path = _write_glue_local_route_log(
                route_log_dir=route_log_dir,
                task=task,
                route="hf_cache_local_files_only",
                path="nyu-mll/glue",
                primary_exc=primary_exc,
                detail="DownloadConfig(local_files_only=True)",
            )
        print(
            f"[dataset] using cached local GLUE dataset for task {task!r}"
            + (f"; audit_log={log_path}" if log_path else ""),
            file=sys.stderr,
        )
        return data, errors
    except Exception as cache_exc:
        errors.append(f"hf_cache_local_files_only: {cache_exc!r}")

    return None, errors


def _write_glue_equivalent_route_log(
        *,
        route_log_dir: str,
        task: str,
        primary_endpoint: str,
        endpoint: str,
        candidate_endpoints,
        revision: str,
        data_files,
        primary_exc: Exception,
        ) -> str:
    os.makedirs(route_log_dir, exist_ok=True)
    log_path = os.path.join(route_log_dir, "glue_dataset_equivalent_route.txt")
    lines = [
        "[glue_dataset_equivalent_route]",
        f"time_utc={datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"task={task}",
        "primary_loader=load_dataset('nyu-mll/glue', task)",
        "equivalent_loader=load_dataset('parquet', data_files=...)",
        f'original_operation=load_dataset("nyu-mll/glue", "{task}")',
        'switched_operation=load_dataset("parquet", data_files=<same GLUE task parquet files>)',
        "route_change_summary=metadata route -> direct parquet file route",
        f"switch_from_endpoint={primary_endpoint}",
        f"switch_to_endpoint={endpoint}",
        f"candidate_endpoints={','.join(candidate_endpoints or [])}",
        f"endpoint={endpoint}",
        f"revision={revision}",
        (
            "semantic_equivalence="
            f"same_repo=nyu-mll/glue; same_task={task}; same_revision={revision}; "
            f"same_splits={','.join(GLUE_PARQUET_SPLITS.get(task, ()))}; "
            "schema_check=required_columns"
        ),
        f"primary_error={type(primary_exc).__name__}: {primary_exc}",
        "data_files:",
    ]
    for split, url in sorted((data_files or {}).items()):
        lines.append(f"  {split}={url}")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")
    return log_path


def load_glue_dataset_equivalent(
        task_name: str,
        *,
        load_dataset_fn=load_dataset,
        load_from_disk_fn=load_from_disk,
        route_log_dir: str = None,
        ):
    task = str(task_name).strip().lower()
    # GLUE data loading has 4 possible routes (HF remote / local save_to_disk /
    # local parquet / HF cache local-only). When debugging "why does this run
    # see stale data?", knowing which route fired is essential — so we log
    # the chosen route to stderr at the point of resolution. The fallback
    # branches already log their own route; here we log when the primary
    # remote loader succeeds.
    try:
        data = load_dataset_fn("nyu-mll/glue", task)
        print(
            f"[dataset] task={task!r} → route=hf_remote endpoint=nyu-mll/glue",
            file=sys.stderr,
        )
        return data
    except Exception as primary_exc:
        if not ENABLE_GLUE_EQUIVALENT_PARQUET_ROUTE:
            raise
        primary_endpoint = _extract_hf_endpoint_from_error(primary_exc)
        revision = _extract_hf_revision_from_error(primary_exc)
        candidate_endpoints = _glue_equivalent_candidate_endpoints(primary_endpoint)
        if _glue_parquet_data_files(task, endpoint=candidate_endpoints[0], revision=revision) is None:
            raise

        data, equivalent_errors = _try_load_local_glue_dataset(
            task,
            load_dataset_fn=load_dataset_fn,
            load_from_disk_fn=load_from_disk_fn,
            route_log_dir=route_log_dir,
            primary_exc=primary_exc,
        )
        if data is not None:
            return data

        for endpoint in candidate_endpoints:
            data_files = _glue_parquet_data_files(task, endpoint=endpoint, revision=revision)
            log_path = None
            if route_log_dir:
                log_path = _write_glue_equivalent_route_log(
                    route_log_dir=route_log_dir,
                    task=task,
                    primary_endpoint=primary_endpoint,
                    endpoint=endpoint,
                    candidate_endpoints=candidate_endpoints,
                    revision=revision,
                    data_files=data_files,
                    primary_exc=primary_exc,
                )
            print(
                "[dataset] load_dataset('nyu-mll/glue', "
                f"{task!r}) failed ({type(primary_exc).__name__}: {primary_exc}); "
                "using equivalent direct parquet route for the same "
                f"nyu-mll/glue revision={revision!r} via {endpoint}"
                + (f"; audit_log={log_path}" if log_path else ""),
                file=sys.stderr,
            )
            try:
                data = load_dataset_fn("parquet", data_files=data_files)
                _validate_glue_dataset_equivalence(data, task)
                return data
            except Exception as equivalent_exc:
                equivalent_errors.append(f"{endpoint}: {equivalent_exc!r}")

        raise RuntimeError(
            f"Failed to load GLUE task {task!r} via nyu-mll/glue and "
            "the local/cache or equivalent parquet routes. "
            f"Primary error: {primary_exc!r}; "
            f"equivalent parquet errors: {equivalent_errors}"
        ) from primary_exc

def parse_bool_flag(raw_value, flag_name):
    if isinstance(raw_value, bool):
        return raw_value
    if raw_value is None:
        return False

    text = str(raw_value).strip().lower()
    if text in ("1", "true", "t", "yes", "y", "on"):
        return True
    if text in ("0", "false", "f", "no", "n", "off", ""):
        return False

    raise ValueError(
        f"Invalid boolean value for {flag_name}: {raw_value!r}. "
        "Expected one of: true/false/1/0/yes/no."
    )


def parse_positive_int(raw_value, flag_name):
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid positive integer for {flag_name}: {raw_value!r}."
        ) from None

    if value <= 0:
        raise ValueError(
            f"Invalid positive integer for {flag_name}: {raw_value!r}."
        )
    return value


def parse_stage1_episode_limit(raw_value, flag_name):
    """Parse Stage-1 episode budget; 0/-1 means unbounded until entropy stop."""
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid integer for {flag_name}: {raw_value!r}."
        ) from None


def parse_optional_positive_float(raw_value, flag_name):
    if raw_value in (None, ""):
        return None
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid positive float for {flag_name}: {raw_value!r}."
        ) from None
    if value <= 0:
        raise ValueError(
            f"Invalid positive float for {flag_name}: {raw_value!r}."
        )
    return value


def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./lora-alpaca",
        adapter_name: str = "lora",
        load_8bit: bool = False,
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.0,
        cutoff_len: int = 256,
        val_set_size: int = 2000,
        use_gradient_checkpointing: bool = False,
        eval_step: int = 200,
        save_step: int = 200,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        # bottleneck adapter hyperparams
        bottleneck_size: int = 256,
        non_linearity: str = "tanh",
        adapter_dropout: float = 0.0,
        use_parallel_adapter: bool = True,
        use_adapterp: bool = False,
        target_modules: List[str] = None,
        # Dora hyperparams
        Wdecompose_target_modules: List[str] = None,
        scaling: Union[float, str] = 1.0,
        # prefix tuning hyperparams
        num_virtual_tokens: int = 30,
        # Sparse tuning hyperparams
        use_ist: bool = False,
        use_rst: bool = False,
        rl_lr: float = 1e-4, 
        stage1_rl_lr: float = None,
        stage2_rl_lr: float = None,
        degree: int = 4,  # degree of polynomial for approximation
        stage1_rl_episodes: int = 51000,
        stage2_rl_episodes: int = 40000,
        stage1_rl_episodes_specified: bool = False,
        stage2_rl_episodes_specified: bool = False,
        stage1_entropy_stop_threshold: float = None,
        ppo_update_interval: int = 120,  # PPO 更新间隔（episode 数）；同时决定 batch 大小与 details 分块大小
        final_eval_config_source: str = "search",  # search | json | manual | max
        final_eval_config_path: str = "glue_final_configs_best_ppo.json",
        manual_stage1_gelu: str = "",
        manual_stage1_softmax: str = "",
        manual_stage2_noise: str = "",
        stage2_fixed_config_source: str = "",
        stage2_fixed_config_path: str = "",
        stage2_manual_gelu: str = "",
        stage2_manual_softmax: str = "",
        final_eval_random_seed: int = 42,
        final_eval_permutation_trials: int = 10,
        final_eval_cost_equivalent_trials: int = 10,
        final_eval_budget_equivalent_trials: int = 10,
        final_eval_stage1_budget_trials: int = 10,
        final_eval_stage2_budget_trials: int = 10,
        final_eval_repeat_n: int = 1,
        final_eval_preset: str = "default",
        final_eval_output_root: str = "",
        final_eval_run_name: str = "",
        final_eval_random_enabled: bool = False,
        final_eval_action_config: str = "",
        final_eval_action_ranges: str = "",
        final_eval_action_fixed: str = "",
        # Same-cost random comparison group for the BLB Stage-2 final eval.
        final_eval_cost_match_count: int = 50,
        final_eval_cost_match_max_attempts: int = 5000,
        # Auto-generate a GLUE benchmark submission zip after final eval.
        final_eval_glue_submission_enabled: bool = True,
        final_eval_glue_submission_seed: int = 42,
        skip_noise_rl: bool = False,
        skip_stage1_rl: bool = False,
        skip_final_eval: bool = False,
        final_eval_only: bool = False,
        resume_run_dir: str = "",
        # 2026-06-01 解耦：新输出布局开关 + stage2-only 的前置 Stage-1 record 选择。
        decoupled_layout: bool = False,
        stage1_run_id: str = "",
        # accuracy constraint params
        stage1_accuracy_tolerance: float = None,
        stage2_limit_tolerance: float = None,
        stage2_stability_tolerance: float = None,
        stage2_k_trials: int = None,
        stage2_probe_size: int = None,
        # Stage-2 RL variant (新版 BLB v3 / 旧版 v2 二选一；默认新版)
        stage2_rl_variant: str = "blb_v3",
        blb_v3_rollout_size: int = None,
        blb_v3_eval_interval: int = None,
        blb_v3_save_interval: int = None,
        blb_v3_calibrate_baseline_samples: int = None,
        blb_v3_rescale_invoker_kind: str = "in_process",
        blb_v3_inproc_rescale_optimizer_root: str = "",
        blb_v3_warmstart_anchor_episodes: int = None,
        blb_v3_warmstart_neighbor_ramp_episodes: int = None,
        blb_v3_warmstart_neighbor_max_mutations: int = None,
        blb_v3_warmstart_neighbor_max_radius: int = None,
        blb_v3_warmstart_neighbor_sampling: bool = None,
        blb_v3_guarded_radius2_enabled: bool = None,
        blb_v3_guarded_radius2_min_episode: int = None,
        blb_v3_guarded_radius2_stall_window: int = None,
        blb_v3_guarded_radius2_max_mutations: int = None,
        blb_v3_guarded_radius2_episode_fraction: float = None,
        blb_v3_guarded_radius2_cooldown_episodes: int = None,
        blb_v3_static_invalid_level_mask_enabled: bool = None,
        blb_v3_warmstart_bias_gain: float = None,
        blb_v3_ent_coef: float = None,
        blb_v3_ent_coef_anchor: float = None,
        blb_v3_ent_coef_ramp_episodes: int = None,
        blb_v3_action_mask_enabled: bool = False,
        blb_v3_action_mask_mode: str = "none",
        blb_v3_action_mask_file: str = "",
        blb_v3_action_mask_baseline_logit_bonus: float = 0.0,
        blb_v3_action_mask_source: str = "",
        # Per-block sequential RL (DEFAULT path since 2026-05-15)
        blb_v3_sequential_rl: bool = True,
        blb_v3_sequential_invalid_penalty: float = 1.0,
        blb_v3_sequential_cost_shaping_coeff: float = 0.05,
        blb_v3_sequential_fusion_shaping_coeff: float = 0.0,
        blb_v3_sequential_early_terminate_on_invalid: bool = False,
        # Multi-seed support (2026-05-16): when None, BLBStage2TrainConfig
        # keeps its default seed=42; when set, overrides so tools/run_multi_seed.sh
        # can sweep N seeds for statistical significance.
        blb_v3_seed: int = None,
        # 2026-05-19: two-GPU reward-probe parallelism. Comma list e.g. "0,1"
        # → BLBStage2Env builds a ProbeRunner that fans K trials across these
        # devices. Empty / single device → single-GPU codepath unchanged.
        blb_v3_reward_devices: str = "",
        # 2026-05-24: Stage-1 RL data-parallel rollout. Comma-separated GPU
        # ids (e.g. "0,1,2,3") → LayerImportanceEvaluator builds a
        # Stage1ParallelRunner that splits the PPO_UPDATE_INTERVAL window
        # across these devices (each worker collects N / num_workers
        # complete episodes per window). Empty / single device → existing
        # single-GPU per-episode loop unchanged.
        stage1_rl_devices: str = "",
        # 2026-06-10: Stage-2 RL episode-parallel rollout (fusion mode only).
        # Comma-separated GPU ids (e.g. "0,1,2,3,4") → N workers each run
        # complete episodes (policy rollout + per-step replan + serial
        # K-trial reward probe) on their own model replica, with global-
        # episode seeding so results are identical for any GPU count.
        # Mutually exclusive with blb_v3_reward_devices. Empty → legacy loop.
        stage2_rl_devices: str = "",
        blb_v3_fast_reward_mode_enabled: bool = False,
        blb_v3_online_k_trials: int = 1,
        blb_v3_terminal_eval_batch_size: int = 4,
        blb_v3_promotion_validation_trials: int = 4,
        blb_v3_promotion_margin_window: float = 0.25,
        # 2026-05-27: 4-sub-stage Stage-2 RL (opt-in). When True, trains one
        # block per sub-stage in --blb_v3_substage_block_order; blocks listed
        # in --blb_v3_substage_frozen_blocks stay at static_skeletons baseline
        # (block 3 by design). See blb_stage2_rl/substage_runner.py.
        blb_v3_substage_mode: bool = False,
        blb_v3_fusion_count_action: bool = False,
        blb_v3_fusion_neighbor_curriculum: bool = True,
        blb_v3_substage_block_order: str = "1,2,4,5",
        blb_v3_substage_frozen_blocks: str = "3",
        blb_v3_substage_episodes_each: int = 15000,
        blb_v3_substage_promotion_top_k: int = 5,
        blb_v3_substage_promotion_trials: int = 8,
        # 2026-05-27: COINN-style OSR pre-prune (opt-in). When osr_results_path
        # is set, the runner either loads existing OSR results from that path
        # or runs a fresh scan saving to that path; the resulting mask is
        # applied alongside the existing 3 masks. osr_scan_only=True exits
        # after the scan (use for the OSR-only preset).
        blb_v3_osr_results_path: str = "",
        blb_v3_osr_scan_only: bool = False,
        blb_v3_osr_num_combo_samples: int = 300,
        blb_v3_osr_allow_fingerprint_mismatch: bool = False,
        # PPO is the only supported RL algorithm. The GRPO experiment path is
        # permanently disabled for this project after the MRPC validation/test
        # mismatch study; keep the argument only so old invocations fail with a
        # clear error instead of silently changing behavior.
        rl_algo: str = "ppo",
        grpo_kl_beta: float = 0.0,
        final_eval_require_rescale_optimizer: bool = False,
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    skip_noise_rl = parse_bool_flag(skip_noise_rl, "skip_noise_rl")
    skip_stage1_rl = parse_bool_flag(skip_stage1_rl, "skip_stage1_rl")
    skip_final_eval = parse_bool_flag(skip_final_eval, "skip_final_eval")
    final_eval_only = parse_bool_flag(final_eval_only, "final_eval_only")
    decoupled_layout = parse_bool_flag(decoupled_layout, "decoupled_layout")
    final_eval_random_enabled = parse_bool_flag(
        final_eval_random_enabled, "final_eval_random_enabled"
    )
    final_eval_require_rescale_optimizer = parse_bool_flag(
        final_eval_require_rescale_optimizer, "final_eval_require_rescale_optimizer"
    )
    rl_algo = str(rl_algo or "ppo").strip().lower()
    if rl_algo != "ppo":
        raise ValueError(
            "GRPO has been disabled for this project after the PPO-vs-GRPO "
            "MRPC generalization study. Use rl_algo='ppo'."
        )
    final_eval_glue_submission_enabled = parse_bool_flag(
        final_eval_glue_submission_enabled, "final_eval_glue_submission_enabled"
    )
    blb_v3_action_mask_enabled = parse_bool_flag(
        blb_v3_action_mask_enabled, "blb_v3_action_mask_enabled"
    )
    blb_v3_sequential_rl = parse_bool_flag(
        blb_v3_sequential_rl, "blb_v3_sequential_rl"
    )
    blb_v3_sequential_early_terminate_on_invalid = parse_bool_flag(
        blb_v3_sequential_early_terminate_on_invalid,
        "blb_v3_sequential_early_terminate_on_invalid",
    )
    blb_v3_fast_reward_mode_enabled = parse_bool_flag(
        blb_v3_fast_reward_mode_enabled,
        "blb_v3_fast_reward_mode_enabled",
    )
    blb_v3_substage_mode = parse_bool_flag(
        blb_v3_substage_mode, "blb_v3_substage_mode"
    )
    blb_v3_fusion_count_action = parse_bool_flag(
        blb_v3_fusion_count_action, "blb_v3_fusion_count_action"
    )
    blb_v3_fusion_neighbor_curriculum = parse_bool_flag(
        blb_v3_fusion_neighbor_curriculum, "blb_v3_fusion_neighbor_curriculum"
    )
    blb_v3_osr_scan_only = parse_bool_flag(
        blb_v3_osr_scan_only, "blb_v3_osr_scan_only"
    )
    blb_v3_osr_allow_fingerprint_mismatch = parse_bool_flag(
        blb_v3_osr_allow_fingerprint_mismatch,
        "blb_v3_osr_allow_fingerprint_mismatch",
    )
    # --final_eval_only 语义：只跑 final eval，不跑任何 RL 搜索阶段。
    # 等价于自动设置 skip_stage1_rl=True & skip_noise_rl=True & skip_final_eval=False，
    # 同时尝试从 resume_run_dir / output_dir 下读取之前搜索得到的最优配置作为 final-eval 输入。
    # 该路径不会安装 graceful-stop 信号、不读写 RL 训练 checkpoint，因此不影响优雅停止与续训。
    if final_eval_only:
        if skip_final_eval:
            raise ValueError(
                "final_eval_only=True 与 skip_final_eval=True 冲突：无可执行项。"
            )
        if not skip_stage1_rl:
            print("[final_eval_only] 自动设置 skip_stage1_rl=True")
            skip_stage1_rl = True
        if not skip_noise_rl:
            print("[final_eval_only] 自动设置 skip_noise_rl=True")
            skip_noise_rl = True
    stage1_rl_episodes_specified = parse_bool_flag(
        stage1_rl_episodes_specified, "stage1_rl_episodes_specified"
    )
    stage2_rl_episodes_specified = parse_bool_flag(
        stage2_rl_episodes_specified, "stage2_rl_episodes_specified"
    )
    batch_size = parse_positive_int(batch_size, "batch_size")
    micro_batch_size = parse_positive_int(micro_batch_size, "micro_batch_size")
    stage1_rl_episodes = parse_stage1_episode_limit(
        stage1_rl_episodes, "stage1_rl_episodes"
    )
    stage2_rl_episodes = parse_positive_int(
        stage2_rl_episodes, "stage2_rl_episodes"
    )
    ppo_update_interval = parse_positive_int(
        ppo_update_interval, "ppo_update_interval"
    )
    stage1_entropy_stop_threshold = parse_optional_positive_float(
        stage1_entropy_stop_threshold, "stage1_entropy_stop_threshold"
    )
    if stage1_rl_episodes <= 0 and stage1_entropy_stop_threshold is None:
        raise ValueError(
            "stage1_rl_episodes <= 0 means unbounded Stage-1 training and "
            "requires stage1_entropy_stop_threshold"
        )
    # 在创建 LayerImportanceEvaluator 之前覆盖 PPO 更新间隔及其派生常量
    import layer_importance_evaluator as _lie
    _lie.set_ppo_update_interval(ppo_update_interval)
    print(
        f"[PPO] ppo_update_interval={_lie.PPO_UPDATE_INTERVAL} "
        f"(batch={_lie.PPO_BATCH_SIZE} steps, details chunk={_lie.STEP_INFO_CHUNK_SIZE} episodes)"
    )

    print(
        f"Finetuning model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"rl_lr: {rl_lr}\n"
        f"stage1_rl_lr: {stage1_rl_lr}\n"
        f"stage2_rl_lr: {stage2_rl_lr}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"Wdecompose_target_modules: {Wdecompose_target_modules}\n"
        f"bottleneck_size: {bottleneck_size}\n"
        f"non_linearity: {non_linearity}\n"
        f"adapter_dropout: {adapter_dropout}\n"
        f"use_parallel_adapter: {use_parallel_adapter}\n"
        f"use_adapterp: {use_adapterp}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"scaling: {scaling}\n"
        f"adapter_name: {adapter_name}\n"
        f"target_modules: {target_modules}\n"
        f"final_eval_config_source: {final_eval_config_source}\n"
        f"final_eval_config_path: {final_eval_config_path}\n"
        f"manual_stage1_gelu: {manual_stage1_gelu}\n"
        f"manual_stage1_softmax: {manual_stage1_softmax}\n"
        f"manual_stage2_noise: {manual_stage2_noise}\n"
        f"stage1_rl_episodes: {stage1_rl_episodes}\n"
        f"stage1_entropy_stop_threshold: {stage1_entropy_stop_threshold}\n"
        f"stage2_rl_episodes: {stage2_rl_episodes}\n"
        f"stage1_rl_episodes_specified: {stage1_rl_episodes_specified}\n"
        f"stage2_rl_episodes_specified: {stage2_rl_episodes_specified}\n"
        f"skip_noise_rl: {skip_noise_rl}\n"
        f"final_eval_repeat_n: {final_eval_repeat_n}\n"
        f"final_eval_preset: {final_eval_preset}\n"
        f"final_eval_output_root: {final_eval_output_root}\n"
        f"final_eval_run_name: {final_eval_run_name}\n"
        f"final_eval_random_enabled: {final_eval_random_enabled}\n"
        f"final_eval_action_config: {final_eval_action_config}\n"
        f"final_eval_action_ranges: {final_eval_action_ranges}\n"
        f"final_eval_action_fixed: {final_eval_action_fixed}\n"
        f"final_eval_require_rescale_optimizer: {final_eval_require_rescale_optimizer}\n"
        f"skip_stage1_rl: {skip_stage1_rl}\n"
        f"skip_final_eval: {skip_final_eval}\n"
        f"final_eval_only: {final_eval_only}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
        f"resume_run_dir: {resume_run_dir}\n"
        f"stage2_rl_variant: {stage2_rl_variant}\n"
        f"blb_v3_rescale_invoker_kind: {blb_v3_rescale_invoker_kind}\n"
        f"blb_v3_inproc_rescale_optimizer_root: {blb_v3_inproc_rescale_optimizer_root}\n"
        f"blb_v3_warmstart_anchor_episodes: {blb_v3_warmstart_anchor_episodes}\n"
        f"blb_v3_warmstart_neighbor_ramp_episodes: {blb_v3_warmstart_neighbor_ramp_episodes}\n"
        f"blb_v3_warmstart_neighbor_max_mutations: {blb_v3_warmstart_neighbor_max_mutations}\n"
        f"blb_v3_warmstart_neighbor_max_radius: {blb_v3_warmstart_neighbor_max_radius}\n"
        f"blb_v3_warmstart_neighbor_sampling: {blb_v3_warmstart_neighbor_sampling}\n"
        f"blb_v3_guarded_radius2_enabled: {blb_v3_guarded_radius2_enabled}\n"
        f"blb_v3_guarded_radius2_min_episode: {blb_v3_guarded_radius2_min_episode}\n"
        f"blb_v3_guarded_radius2_stall_window: {blb_v3_guarded_radius2_stall_window}\n"
        f"blb_v3_guarded_radius2_max_mutations: {blb_v3_guarded_radius2_max_mutations}\n"
        f"blb_v3_guarded_radius2_episode_fraction: {blb_v3_guarded_radius2_episode_fraction}\n"
        f"blb_v3_guarded_radius2_cooldown_episodes: {blb_v3_guarded_radius2_cooldown_episodes}\n"
        f"blb_v3_static_invalid_level_mask_enabled: {blb_v3_static_invalid_level_mask_enabled}\n"
        f"blb_v3_warmstart_bias_gain: {blb_v3_warmstart_bias_gain}\n"
        f"blb_v3_ent_coef: {blb_v3_ent_coef}\n"
        f"blb_v3_ent_coef_anchor: {blb_v3_ent_coef_anchor}\n"
        f"blb_v3_ent_coef_ramp_episodes: {blb_v3_ent_coef_ramp_episodes}\n"
        f"blb_v3_action_mask_enabled: {blb_v3_action_mask_enabled}\n"
        f"blb_v3_action_mask_mode: {blb_v3_action_mask_mode}\n"
        f"blb_v3_action_mask_file: {blb_v3_action_mask_file}\n"
        f"blb_v3_action_mask_baseline_logit_bonus: {blb_v3_action_mask_baseline_logit_bonus}\n"
        f"blb_v3_action_mask_source: {blb_v3_action_mask_source}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    run_output_dir = str(output_dir or "").strip()
    trainer_output_dir = (
        os.path.join(run_output_dir, "trainer_output")
        if run_output_dir
        else "./inference_output"
    )
    os.makedirs(trainer_output_dir, exist_ok=True)
    seed_everything(final_eval_random_seed)

    # device_map = "gpu"
    ddp = True  # Distributed Data Parallelism disabled

    device_map = "cuda"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    #     gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if 'llama' in base_model and 'llama3' not in base_model:
        # Due to the name of transformers' LlamaTokenizer, we have to do this
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)


    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"

    if load_8bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
            quantization_config=quantization_config,
        )
    else:
        config = AutoConfig.from_pretrained(base_model)
        # config.use_causal_lm = False  # Key: disable causal mask for MRPC.
        _dp = data_path.lower()
        if _dp == "stsb":
            _num_labels = 1
        elif _dp == "mnli":
            _num_labels = 3
        else:
            _num_labels = 2
        print(f"Auto-detected num_labels={_num_labels} for dataset '{data_path}'")

        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=_num_labels,
            # load_in_8bit=False,
            # torch_dtype=torch.float16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            # device_map ="cpu",
            trust_remote_code=True,
            # pad_token_id=tokenizer.eos_token_id
            pad_token_id=tokenizer.pad_token_id,
        )

    # ---------------------------------------------------------------
    # Freeze the backbone. The downstream pipeline (layer_importance_
    # evaluator.py + noise_rl_module_v2.py) only uses this HF model for
    # **inference** to compute rewards — the PPO policy/value networks
    # are the only thing being trained. Explicitly disabling grads on
    # every parameter and pinning the model to eval() makes that
    # contract bulletproof: no amount of noise-wrapping, function-
    # replacement or stray autograd call can push an update into the
    # pretrained weights, and dropout/train-mode side-effects cannot
    # add variance to the reward signal mid-episode.
    # ---------------------------------------------------------------
    for _param in model.parameters():
        _param.requires_grad_(False)
    model.eval()

    model.to("cuda")


    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in base_model:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in base_model:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

    # Tokenize helper.
    def tokenize_fn(examples):
        _dp = data_path.lower()
        if _dp in ("sst2", "cola"):
            tokenized = tokenizer(
                examples["sentence"],
                truncation=True, padding=False, max_length=128, return_tensors=None,
            )
        elif _dp == "qnli":
            tokenized = tokenizer(
                examples["question"],
                examples["sentence"],
                truncation=True, padding=False, max_length=128, return_tensors=None,
            )
        elif _dp == "mnli":
            tokenized = tokenizer(
                examples["premise"],
                examples["hypothesis"],
                truncation=True, padding=False, max_length=128, return_tensors=None,
            )
        else:  # mrpc, stsb, rte, wnli
            tokenized = tokenizer(
                examples["sentence1"],
                examples["sentence2"],
                truncation=True, padding=False, max_length=128, return_tensors=None,
            )
        return tokenized

    # def generate_and_tokenize_prompt(data_point):
    #     full_prompt = generate_prompt(data_point)
    #     tokenized_full_prompt = tokenize(full_prompt)
    #     if not train_on_inputs:
    #         user_prompt = generate_prompt({**data_point, "output": ""})
    #         tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
    #         user_prompt_len = len(tokenized_user_prompt["input_ids"])

    #         tokenized_full_prompt["labels"] = [
    #                                               -100
    #                                           ] * user_prompt_len + tokenized_full_prompt["labels"][
    #                                                                 user_prompt_len:
    #                                                                 ]  # could be sped up, probably
    #     return tokenized_full_prompt

    # model = prepare_model_for_int8_training(model, use_gradient_checkpointing=use_gradient_checkpointing)

    # if adapter_name == "lora":
    #     config = LoraConfig(
    #         r=lora_r,
    #         lora_alpha=lora_alpha,
    #         target_modules=target_modules,
    #         lora_dropout=lora_dropout,
    #         bias="none",
    #         task_type="CAUSAL_LM",
    #     )
    # elif adapter_name == "dora":
    #     dora_simple = True
    #     config = DoraConfig(
    #         r=lora_r,
    #         lora_alpha=lora_alpha,
    #         target_modules=target_modules,
    #         lora_dropout=lora_dropout,
    #         bias="none",
    #         task_type="CAUSAL_LM",
    #         dora_simple=dora_simple,
    #         Wdecompose_target_modules=Wdecompose_target_modules
    #     )
    # elif adapter_name == "bottleneck":
    #     config = BottleneckConfig(
    #         bottleneck_size=bottleneck_size,
    #         non_linearity=non_linearity,
    #         adapter_dropout=adapter_dropout,
    #         use_parallel_adapter=use_parallel_adapter,
    #         use_adapterp=use_adapterp,
    #         target_modules=target_modules,
    #         scaling=scaling,
    #         bias="none",
    #         task_type="CAUSAL_LM",
    #     )
    # elif adapter_name == "prefix-tuning":
    #     config = PrefixTuningConfig(
    #         num_virtual_tokens=num_virtual_tokens,
    #         task_type="CAUSAL_LM",
    #     )
    # model = get_peft_model(model, config)

    if adapter_name == "prefix-tuning":
        model.to("cuda") 
    
    print(model)
    if data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        # glue tasks: "stsb", "mnli", "sst2", "cola", "qnli", "rte", "wnli", "mrpc"
        data = load_glue_dataset_equivalent(
            data_path,
            route_log_dir=os.path.join(output_dir, "logs"),
        )



    
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            # adapters_weights = torch.load(checkpoint_name)
            # model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    # model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    # if val_set_size > 0:
    #     train_val = data["train"].train_test_split(
    #         test_size=val_set_size, shuffle=True, seed=42
    #     )
    #     train_data = (
    #         train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    #     )
    #     val_data = (
    #         train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    #     )
    # else:
    #     train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    #     val_data = None
    
    # MNLI needs special handling: matched and mismatched validation splits.
    val_data_mm = None  # MNLI mismatched validation split.
    
    if val_set_size > 0:
        is_mnli = data_path.lower() == 'mnli'
        
        if is_mnli:
            print(f"Loading MNLI dataset (matched + mismatched validation sets)")
            train_data = data["train"].shuffle(seed=final_eval_random_seed).map(tokenize_fn)
            val_data = data["validation_matched"].shuffle(seed=final_eval_random_seed).map(tokenize_fn)
            val_data_mm = data["validation_mismatched"].shuffle(seed=final_eval_random_seed).map(tokenize_fn)
            
            print(f"After tokenize matched: {val_data[0]}")
            train_data = train_data.rename_column("label", "labels")
            val_data = val_data.rename_column("label", "labels")
            val_data_mm = val_data_mm.rename_column("label", "labels")
            
            columns = ["input_ids", "attention_mask", "token_type_ids", "labels"]
            train_data.set_format(type="torch", columns=columns)
            val_data.set_format(type="torch", columns=columns)
            val_data_mm.set_format(type="torch", columns=columns)
            
            print(f"Train data size: {len(train_data)}")
            print(f"Validation matched size: {len(val_data)}")
            print(f"Validation mismatched size: {len(val_data_mm)}")
        else:
            print(f"Loading dataset: {data['validation']}")
            train_data = data["train"].shuffle(seed=final_eval_random_seed).map(tokenize_fn)
            val_data = data["validation"].shuffle(seed=final_eval_random_seed).map(tokenize_fn)
            # The current RL flow does not use the official test split.
            # test_data = data["test"].shuffle().map(tokenize_fn)
            
            print(f"After tokenize: {val_data[0]}")
            # add label
            train_data = train_data.rename_column("label", "labels")
            val_data = val_data.rename_column("label", "labels")
            
            print(f"After add label: {val_data[0]}")
            
            # Set PyTorch tensor format.
            columns = ["input_ids", "attention_mask", "token_type_ids", "labels"]
            train_data.set_format(type="torch", columns=columns)
            val_data.set_format(type="torch", columns=columns)

            print(f"After format: {val_data}")
            
            print(f"Train data size: {len(train_data)}")
            print(f"Validation data size: {len(val_data)}") 
            # print(f"Test data size: {len(test_data)}")
            
    else:
        train_data = data["train"].shuffle(seed=final_eval_random_seed).map(tokenize_fn)
        val_data = None

    # data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

    # for Binary classification task
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding= "max_length",
        max_length=128,     # Effective when padding="max_length"
        return_tensors="pt", # Return PyTorch tensors
        pad_to_multiple_of=8   # Return attention masks
    )
    
    # if not ddp and torch.cuda.device_count() > 1:
    #     # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    #     model.is_parallelizable = True
    #     model.model_parallel = True
    parsed_manual_stage1_gelu = parse_degree_config(manual_stage1_gelu)
    parsed_manual_stage1_softmax = parse_degree_config(manual_stage1_softmax)
    parsed_manual_stage2_noise = parse_noise_config(manual_stage2_noise)
    trainer_callbacks = []

    if use_ist:
        from layer_importance_evaluator import LayerImportanceEvaluator
        print('Reinforcement Learning to evaluate layer sensitivity to approximation')
        # Pass data_path so evaluator can detect dataset type and metrics.
        importance_evaluator = LayerImportanceEvaluator(
            model=model, 
            train_data=train_data, 
            # Keep the historical argument name; we pass validation data here.
            test_data=val_data, 
            data_collator=data_collator, 
            batch_size=batch_size,
            rl_lr=rl_lr, 
            stage1_rl_lr=stage1_rl_lr,
            stage2_rl_lr=stage2_rl_lr,
            degree=degree,
            stage1_rl_episodes=stage1_rl_episodes,
            stage2_rl_episodes=stage2_rl_episodes,
            stage1_rl_episodes_specified=stage1_rl_episodes_specified,
            stage2_rl_episodes_specified=stage2_rl_episodes_specified,
            stage1_entropy_stop_threshold=stage1_entropy_stop_threshold,
            run_output_dir=run_output_dir,
            final_eval_config_source=final_eval_config_source,
            final_eval_config_path=final_eval_config_path,
            manual_stage1_gelu=parsed_manual_stage1_gelu,
            manual_stage1_softmax=parsed_manual_stage1_softmax,
            manual_stage2_noise=parsed_manual_stage2_noise,
            final_eval_random_seed=final_eval_random_seed,
            final_eval_permutation_trials=final_eval_permutation_trials,
            final_eval_cost_equivalent_trials=final_eval_cost_equivalent_trials,
            final_eval_budget_equivalent_trials=final_eval_budget_equivalent_trials,
            final_eval_stage1_budget_trials=final_eval_stage1_budget_trials,
            final_eval_stage2_budget_trials=final_eval_stage2_budget_trials,
            final_eval_repeat_n=final_eval_repeat_n,
            final_eval_preset=final_eval_preset,
            final_eval_output_root=final_eval_output_root,
            final_eval_run_name=final_eval_run_name,
            final_eval_random_enabled=final_eval_random_enabled,
            final_eval_action_config=final_eval_action_config,
            final_eval_action_ranges=final_eval_action_ranges,
            final_eval_action_fixed=final_eval_action_fixed,
            final_eval_cost_match_count=final_eval_cost_match_count,
            final_eval_cost_match_max_attempts=final_eval_cost_match_max_attempts,
            final_eval_glue_submission_enabled=final_eval_glue_submission_enabled,
            final_eval_glue_submission_seed=final_eval_glue_submission_seed,
            final_eval_require_rescale_optimizer=final_eval_require_rescale_optimizer,
            skip_noise_rl=skip_noise_rl,
            skip_stage1_rl=skip_stage1_rl,
            skip_final_eval=skip_final_eval,
            final_eval_only=final_eval_only,
            resume_run_dir=resume_run_dir,
            decoupled_layout=decoupled_layout,
            stage1_run_id=stage1_run_id,
            data_path=data_path,
            test_data_mm=val_data_mm,
            stage1_accuracy_tolerance=stage1_accuracy_tolerance,
            stage2_limit_tolerance=stage2_limit_tolerance,
            stage2_stability_tolerance=stage2_stability_tolerance,
            stage2_k_trials=stage2_k_trials,
            stage2_probe_size=stage2_probe_size,
            stage2_rl_variant=stage2_rl_variant,
            blb_v3_rescale_invoker_kind=blb_v3_rescale_invoker_kind,
            blb_v3_inproc_rescale_optimizer_root=(
                blb_v3_inproc_rescale_optimizer_root
                if blb_v3_inproc_rescale_optimizer_root not in (None, "") else None
            ),
            blb_v3_rollout_size=blb_v3_rollout_size,
            blb_v3_eval_interval=blb_v3_eval_interval,
            blb_v3_save_interval=blb_v3_save_interval,
            blb_v3_calibrate_baseline_samples=blb_v3_calibrate_baseline_samples,
            blb_v3_warmstart_anchor_episodes=blb_v3_warmstart_anchor_episodes,
            blb_v3_warmstart_neighbor_ramp_episodes=blb_v3_warmstart_neighbor_ramp_episodes,
            blb_v3_warmstart_neighbor_max_mutations=blb_v3_warmstart_neighbor_max_mutations,
            blb_v3_warmstart_neighbor_max_radius=blb_v3_warmstart_neighbor_max_radius,
            blb_v3_warmstart_neighbor_sampling=blb_v3_warmstart_neighbor_sampling,
            blb_v3_guarded_radius2_enabled=blb_v3_guarded_radius2_enabled,
            blb_v3_guarded_radius2_min_episode=blb_v3_guarded_radius2_min_episode,
            blb_v3_guarded_radius2_stall_window=blb_v3_guarded_radius2_stall_window,
            blb_v3_guarded_radius2_max_mutations=blb_v3_guarded_radius2_max_mutations,
            blb_v3_guarded_radius2_episode_fraction=blb_v3_guarded_radius2_episode_fraction,
            blb_v3_guarded_radius2_cooldown_episodes=blb_v3_guarded_radius2_cooldown_episodes,
            blb_v3_static_invalid_level_mask_enabled=blb_v3_static_invalid_level_mask_enabled,
            blb_v3_warmstart_bias_gain=blb_v3_warmstart_bias_gain,
            blb_v3_ent_coef=blb_v3_ent_coef,
            blb_v3_ent_coef_anchor=blb_v3_ent_coef_anchor,
            blb_v3_ent_coef_ramp_episodes=blb_v3_ent_coef_ramp_episodes,
            blb_v3_action_mask_enabled=blb_v3_action_mask_enabled,
            blb_v3_action_mask_mode=blb_v3_action_mask_mode,
            blb_v3_action_mask_file=blb_v3_action_mask_file,
            blb_v3_action_mask_baseline_logit_bonus=blb_v3_action_mask_baseline_logit_bonus,
            blb_v3_action_mask_source=blb_v3_action_mask_source,
            blb_v3_sequential_rl=blb_v3_sequential_rl,
            blb_v3_sequential_invalid_penalty=blb_v3_sequential_invalid_penalty,
            blb_v3_sequential_cost_shaping_coeff=blb_v3_sequential_cost_shaping_coeff,
            blb_v3_sequential_fusion_shaping_coeff=blb_v3_sequential_fusion_shaping_coeff,
            blb_v3_sequential_early_terminate_on_invalid=blb_v3_sequential_early_terminate_on_invalid,
            blb_v3_seed=blb_v3_seed,
            blb_v3_reward_devices=blb_v3_reward_devices,
            stage1_rl_devices=stage1_rl_devices,
            stage2_rl_devices=stage2_rl_devices,
            blb_v3_fast_reward_mode_enabled=blb_v3_fast_reward_mode_enabled,
            blb_v3_online_k_trials=blb_v3_online_k_trials,
            blb_v3_terminal_eval_batch_size=blb_v3_terminal_eval_batch_size,
            blb_v3_promotion_validation_trials=blb_v3_promotion_validation_trials,
            blb_v3_promotion_margin_window=blb_v3_promotion_margin_window,
            blb_v3_substage_mode=blb_v3_substage_mode,
            blb_v3_fusion_count_action=blb_v3_fusion_count_action,
            blb_v3_fusion_neighbor_curriculum=blb_v3_fusion_neighbor_curriculum,
            blb_v3_substage_block_order=blb_v3_substage_block_order,
            blb_v3_substage_frozen_blocks=blb_v3_substage_frozen_blocks,
            blb_v3_substage_episodes_each=blb_v3_substage_episodes_each,
            blb_v3_substage_promotion_top_k=blb_v3_substage_promotion_top_k,
            blb_v3_substage_promotion_trials=blb_v3_substage_promotion_trials,
            blb_v3_osr_results_path=blb_v3_osr_results_path,
            blb_v3_osr_scan_only=blb_v3_osr_scan_only,
            blb_v3_osr_num_combo_samples=blb_v3_osr_num_combo_samples,
            blb_v3_osr_allow_fingerprint_mismatch=blb_v3_osr_allow_fingerprint_mismatch,
            rl_algo=rl_algo,
            grpo_kl_beta=grpo_kl_beta,
        )
        trainer_callbacks.append(importance_evaluator)
    # elif use_rst:
    #     from rst import RSTCallback
    #     print('Random Sparse Tuning activated')
    #     rst_callback = RSTCallback(model)
    #     trainer_callbacks.append(rst_callback)
    else:
        print('No sparse tuning activated')
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            output_dir=trainer_output_dir,
            per_device_eval_batch_size=batch_size,  # 推理批次大小
            disable_tqdm=False,  # Optional progress bar control
            # per_device_train_batch_size=micro_batch_size,
            # gradient_accumulation_steps=gradient_accumulation_steps,
            # warmup_steps=100,
            # num_train_epochs=num_epochs,
            # learning_rate=learning_rate,
            # weight_decay=weight_decay,
            # # fp16=True,
            # fp16=False,
            # fp16_full_eval=False,
            # logging_steps=10,
            # optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            # save_strategy="steps",
            eval_steps=eval_step if val_set_size > 0 else None,
            # save_steps=save_step,
            # output_dir=output_dir,
            # save_total_limit=3,
            # load_best_model_at_end=True if val_set_size > 0 else False,
            # ddp_find_unused_parameters=False if ddp else None,
            # group_by_length=group_by_length,
            # report_to="wandb" if use_wandb else None,
            # run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=data_collator,
        callbacks=trainer_callbacks
    )

    model.config.use_cache = False
    model.config.is_decoder = False

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    print( "Model compile started")
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    
    # trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # model.save_pretrained(output_dir)

    print("Model compile finished")
    print("Starting evaluation...")

    # trainer.predict(test_dataset=val_data, metric_key_prefix="predict")

    for _ in range(1):
        print(f"Round {_} of evaluation")
        
        print(val_data[0])  # Should be list[int]
        print(val_data[0])   # Should be consistent

        eval_results = trainer.evaluate(eval_dataset=val_data)
        final_loss = eval_results["eval_loss"] if "eval_loss" in eval_results else None
        print(f"Round {_}, Final evaluation loss: {final_loss}")

    
    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}

                ### Input:
                {data_point["input"]}

                ### Response:
                {data_point["output"]}"""  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}

                ### Response:
                {data_point["output"]}"""  # noqa: E501


if __name__ == "__main__":
    run_fire_entrypoint(
        fire,
        train,
        program_name="rl_tune.py",
    )
