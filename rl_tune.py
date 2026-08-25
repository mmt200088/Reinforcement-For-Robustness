import os
import sys
import json
import re
import glob
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fire
import torch
import transformers
from datasets import DownloadConfig, load_dataset, load_from_disk
from rfr.common.runtime_error_reporter import run_fire_entrypoint


from transformers import (  # noqa: F402
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from rfr.common.cli_parse_utils import (
    parse_bool_flag,
    parse_degree_config,
    parse_noise_config,
    parse_optional_positive_float,
    parse_positive_int,
    parse_stage1_episode_limit,
    parse_stage2_episode_limit,
)
from rfr.preparation.data.protocol import (
    GLUE_DATASET_REVISION,
    GlueDataProtocolContext,
    load_train_probe_fixture,
    resolve_glue_protocol_views,
    resolve_model_family,
    validate_dataset,
    validate_supported_profile,
)
from rfr.preparation.data.mrpc_reproducibility import (
    MRPC_FULL_EXAMPLE_COUNT,
    MRPCReproducibilityContext,
    MRPCReproducibilityError,
    load_mrpc_fixture,
    resolve_mrpc_pretrained_revision_kwargs,
    resolve_mrpc_validation_views,
)


ENABLE_GLUE_EQUIVALENT_PARQUET_ROUTE = True
MRPC_VALIDATION_ROW_COUNT = MRPC_FULL_EXAMPLE_COUNT
GLUE_EQUIVALENT_PARQUET_ENDPOINTS = [
    "https://huggingface.co",
]
GLUE_LOCAL_DATASET_ENV_VARS = (
    "GLUE_LOCAL_DATASET_DIR",
    "GLUE_DATASET_DIR",
)


def resolve_pretrained_revision_kwargs(
        *,
        fixture,
        data_path: str,
        model_id: str,
        ) -> tuple[dict[str, str], dict[str, str]]:
    return resolve_mrpc_pretrained_revision_kwargs(
        fixture=fixture,
        data_path=data_path,
        model_id=model_id,
    )


def seed_everything(seed: int) -> int:
    seed = int(seed)
    transformers.set_seed(seed)
    try:
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    return seed


GLUE_PARQUET_SPLITS = {
    "mrpc": ("train", "validation", "test"),
    "rte": ("train", "validation", "test"),
    "sst2": ("train", "validation", "test"),
}

GLUE_REQUIRED_COLUMNS = {
    "mrpc": ("sentence1", "sentence2", "label", "idx"),
    "rte": ("sentence1", "sentence2", "label", "idx"),
    "sst2": ("sentence", "label", "idx"),
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


def resolve_mrpc_reproducibility_views(
        data: Any,
        *,
        data_path: str,
        fixture,
        ):
    if fixture is None:
        raise MRPCReproducibilityError(
            "MRPC reproducibility requires a fixture"
        )
    if str(data_path).strip().lower() != "mrpc":
        raise MRPCReproducibilityError(
            "MRPC reproducibility fixture requires the MRPC task"
        )
    return resolve_mrpc_validation_views(
        data,
        fixture,
        expected_row_count=MRPC_VALIDATION_ROW_COUNT,
    )


def _finalize_glue_load_success(
        data,
        task_name: str,
        *,
        mrpc_reproducibility_fixture=None,
        ):
    task = str(task_name).strip().lower()
    try:
        _validate_glue_dataset_equivalence(data, task)
    except Exception as exc:
        if mrpc_reproducibility_fixture is not None:
            raise MRPCReproducibilityError(
                f"MRPC GLUE {task!r} schema mismatch"
            ) from exc
        raise

    if mrpc_reproducibility_fixture is None:
        return data
    if task != "mrpc":
        raise MRPCReproducibilityError(
            "MRPC reproducibility fixture cannot validate another GLUE task"
        )
    resolve_mrpc_validation_views(
        data,
        mrpc_reproducibility_fixture,
        expected_row_count=MRPC_VALIDATION_ROW_COUNT,
    )
    return data


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
        mrpc_reproducibility_fixture=None,
        ):
    errors = []
    for path in _glue_local_dataset_candidates(task):
        if not os.path.isdir(path):
            continue
        try:
            data = load_from_disk_fn(path)
            _finalize_glue_load_success(
                data,
                task,
                mrpc_reproducibility_fixture=mrpc_reproducibility_fixture,
            )
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
        except MRPCReproducibilityError:
            raise
        except Exception as local_exc:
            errors.append(f"{path} load_from_disk: {local_exc!r}")

        data_files = _glue_local_parquet_data_files(task, path)
        if data_files is None:
            continue
        try:
            data = load_dataset_fn("parquet", data_files=data_files)
            _finalize_glue_load_success(
                data,
                task,
                mrpc_reproducibility_fixture=mrpc_reproducibility_fixture,
            )
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
        except MRPCReproducibilityError:
            raise
        except Exception as local_exc:
            errors.append(f"{path} local_parquet: {local_exc!r}")

    try:
        data = load_dataset_fn(
            "nyu-mll/glue",
            task,
            revision=GLUE_DATASET_REVISION,
            download_config=DownloadConfig(local_files_only=True),
        )
        _finalize_glue_load_success(
            data,
            task,
            mrpc_reproducibility_fixture=mrpc_reproducibility_fixture,
        )
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
    except MRPCReproducibilityError:
        raise
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
        mrpc_reproducibility_fixture=None,
        ):
    task = str(task_name).strip().lower()
    validate_dataset(task)


    try:
        data = load_dataset_fn(
            "nyu-mll/glue",
            task,
            revision=GLUE_DATASET_REVISION,
        )
        _finalize_glue_load_success(
            data,
            task,
            mrpc_reproducibility_fixture=mrpc_reproducibility_fixture,
        )
        print(
            f"[dataset] task={task!r} → route=hf_remote endpoint=nyu-mll/glue",
            file=sys.stderr,
        )
        return data
    except MRPCReproducibilityError:
        raise
    except Exception as primary_exc:
        if not ENABLE_GLUE_EQUIVALENT_PARQUET_ROUTE:
            raise
        primary_endpoint = _extract_hf_endpoint_from_error(primary_exc)
        revision = GLUE_DATASET_REVISION
        candidate_endpoints = _glue_equivalent_candidate_endpoints(primary_endpoint)
        if _glue_parquet_data_files(task, endpoint=candidate_endpoints[0], revision=revision) is None:
            raise

        data, equivalent_errors = _try_load_local_glue_dataset(
            task,
            load_dataset_fn=load_dataset_fn,
            load_from_disk_fn=load_from_disk_fn,
            route_log_dir=route_log_dir,
            primary_exc=primary_exc,
            mrpc_reproducibility_fixture=mrpc_reproducibility_fixture,
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
                _finalize_glue_load_success(
                    data,
                    task,
                    mrpc_reproducibility_fixture=mrpc_reproducibility_fixture,
                )
                return data
            except MRPCReproducibilityError:
                raise
            except Exception as equivalent_exc:
                equivalent_errors.append(f"{endpoint}: {equivalent_exc!r}")

        raise RuntimeError(
            f"Failed to load GLUE task {task!r} via nyu-mll/glue and "
            "the local/cache or equivalent parquet routes. "
            f"Primary error: {primary_exc!r}; "
            f"equivalent parquet errors: {equivalent_errors}"
        ) from primary_exc

def train(
        base_model: str = "",
        data_path: str = "mrpc",
        output_dir: str = "./runs",
        glue_train_probe_fixture_path: str = "fixtures/reproducibility/glue_train_probe_v1.json",
        mrpc_reproducibility_fixture_path: str = "",
        batch_size: int = 128,
        rl_lr: float = 1e-4,
        stage1_rl_lr: float = None,
        stage2_rl_lr: float = None,
        degree: int = 4,
        stage1_rl_episodes: int = 51000,
        stage2_rl_episodes: int = 0,
        stage1_rl_episodes_specified: bool = False,
        stage2_rl_episodes_specified: bool = False,
        stage1_entropy_stop_threshold: float = None,
        ppo_update_interval: int = 120,
        final_eval_config_source: str = "search",
        final_eval_config_path: str = "glue_final_configs_best_ppo.json",
        manual_stage1_gelu: str = "",
        manual_stage1_softmax: str = "",
        manual_stage2_noise: str = "",
        stage2_fixed_config_source: str = "all4",
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

        final_eval_cost_match_count: int = 50,
        final_eval_cost_match_max_attempts: int = 5000,
        skip_noise_rl: bool = False,
        skip_stage1_rl: bool = False,
        skip_final_eval: bool = False,
        final_eval_only: bool = False,
        resume_run_dir: str = "",

        decoupled_layout: bool = False,
        stage1_run_id: str = "",

        stage1_accuracy_tolerance: float = None,
        stage2_limit_tolerance: float = None,
        stage2_stability_tolerance: float = None,
        stage2_stability_multiplier: float = 2.0,
        stage2_communication_importance_ratio: float = 1.0,
        stage2_k_trials: int = None,
        stage2_probe_size: int = None,
        stage2_inference_batch_size: int = None,

        blb_v3_rollout_size: int = None,
        blb_v3_eval_interval: int = None,
        blb_v3_save_interval: int = None,
        blb_v3_calibrate_baseline_samples: int = None,
        blb_v3_inproc_rescale_optimizer_root: str = "",
        blb_v3_seed: int = None,
        blb_v3_reward_devices: str = "",
        stage1_rl_devices: str = "",
        blb_v3_online_k_trials: int = 3,
        blb_v3_terminal_eval_batch_size: int = 4,
        blb_v3_promotion_validation_trials: int = 15,
        blb_v3_final_selection_top_n: int = 20,
        blb_v3_final_selection_validation_trials: int = 15,
        blb_v3_promotion_margin_window: float = 0.25,
        blb_v3_baseline_groups: int = 5,
        blb_v3_baseline_trials_per_group: int = 3,
        blb_v3_constraint_bootstrap_samples: int = 4096,
        blb_v3_online_constraint_probability: float = 0.50,
        blb_v3_promotion_constraint_probability: float = 0.80,
        blb_v3_final_constraint_probability: float = 0.95,
        blb_v3_min_convergence_episodes: int = 90_000,
        blb_v3_convergence_patience_updates: int = 100,
        blb_v3_search_backend: str = "ppo",
        blb_v3_search_evaluation_budget: int = 0,
        blb_v3_search_initial_design_size: int = 64,
        blb_v3_search_candidate_pool_size: int = 2048,
        blb_v3_search_population_size: int = 64,
        blb_v3_search_patience_generations: int = 100,
        blb_v3_search_mutation_max_coordinates: int = 3,
        blb_v3_search_rf_n_estimators: int = 128,
        blb_v3_search_rf_min_samples_leaf: int = 2,
        blb_v3_search_full_validation: bool = True,
        comparator_smoke: bool = False,
        comparator_stage1_only: bool = False,
):
    data_path = validate_dataset(data_path)
    model_family = resolve_model_family(base_model)
    validate_supported_profile(model_family, data_path)

    skip_noise_rl = parse_bool_flag(skip_noise_rl, "skip_noise_rl")
    skip_stage1_rl = parse_bool_flag(skip_stage1_rl, "skip_stage1_rl")
    skip_final_eval = parse_bool_flag(skip_final_eval, "skip_final_eval")
    final_eval_only = parse_bool_flag(final_eval_only, "final_eval_only")
    decoupled_layout = parse_bool_flag(decoupled_layout, "decoupled_layout")
    final_eval_random_enabled = parse_bool_flag(
        final_eval_random_enabled, "final_eval_random_enabled"
    )
    from blb_stage2_rl.search_baselines import normalize_search_backend

    blb_v3_search_backend = normalize_search_backend(
        blb_v3_search_backend
    )
    blb_v3_search_full_validation = parse_bool_flag(
        blb_v3_search_full_validation,
        "blb_v3_search_full_validation",
    )
    comparator_smoke = parse_bool_flag(
        comparator_smoke, "comparator_smoke",
    )
    comparator_stage1_only = parse_bool_flag(
        comparator_stage1_only, "comparator_stage1_only",
    )
    stage2_stability_multiplier = float(stage2_stability_multiplier)
    if stage2_stability_multiplier <= 0.0:
        raise ValueError("stage2_stability_multiplier must be positive")
    from rfr.search.common.precision_presets import (
        validate_communication_importance_ratio,
    )
    stage2_communication_importance_ratio = (
        validate_communication_importance_ratio(
            stage2_communication_importance_ratio,
        )
    )
    for name, value in (
            ("blb_v3_baseline_groups", blb_v3_baseline_groups),
            ("blb_v3_baseline_trials_per_group", blb_v3_baseline_trials_per_group),
            ("blb_v3_constraint_bootstrap_samples", blb_v3_constraint_bootstrap_samples),
            ("blb_v3_promotion_validation_trials", blb_v3_promotion_validation_trials),
            ("blb_v3_final_selection_validation_trials", blb_v3_final_selection_validation_trials),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (
            ("blb_v3_online_constraint_probability", blb_v3_online_constraint_probability),
            ("blb_v3_promotion_constraint_probability", blb_v3_promotion_constraint_probability),
            ("blb_v3_final_constraint_probability", blb_v3_final_constraint_probability),
    ):
        probability = float(value)
        if not 0.0 < probability <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")
    if not (
            float(blb_v3_online_constraint_probability)
            <= float(blb_v3_promotion_constraint_probability)
            <= float(blb_v3_final_constraint_probability)
    ):
        raise ValueError(
            "constraint probabilities must satisfy online <= promotion <= final"
        )
    if int(blb_v3_min_convergence_episodes) < 90_000:
        raise ValueError(
            "blb_v3_min_convergence_episodes must be at least 90000"
        )
    if int(blb_v3_convergence_patience_updates) < 100:
        raise ValueError(
            "blb_v3_convergence_patience_updates must be at least 100"
        )


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
    if stage2_inference_batch_size in (None, ""):
        stage2_inference_batch_size = None
    else:
        stage2_inference_batch_size = parse_positive_int(
            stage2_inference_batch_size, "stage2_inference_batch_size",
        )
    stage1_rl_episodes = parse_stage1_episode_limit(
        stage1_rl_episodes, "stage1_rl_episodes"
    )
    stage2_rl_episodes = parse_stage2_episode_limit(
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

    glue_fixture_path = Path(
        str(glue_train_probe_fixture_path or "").strip()
    ).expanduser()
    if not glue_fixture_path.is_absolute():
        glue_fixture_path = Path(__file__).resolve().parent / glue_fixture_path
    glue_fixture = load_train_probe_fixture(glue_fixture_path)

    mrpc_fixture = None
    mrpc_views = None
    mrpc_reproducibility = None
    fixture_path = str(
        mrpc_reproducibility_fixture_path or ""
    ).strip()
    if fixture_path:
        mrpc_fixture = load_mrpc_fixture(fixture_path)
    model_revision_kwargs, tokenizer_revision_kwargs = (
        resolve_pretrained_revision_kwargs(
            fixture=mrpc_fixture,
            data_path=data_path,
            model_id=base_model,
        )
    )


    import layer_importance_evaluator as _lie
    _lie.set_ppo_update_interval(ppo_update_interval)
    print(
        f"[PPO] ppo_update_interval={_lie.PPO_UPDATE_INTERVAL} "
        f"(batch={_lie.PPO_BATCH_SIZE} steps, details chunk={_lie.STEP_INFO_CHUNK_SIZE} episodes)"
    )

    print(
        "Running approximation search with parameters:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"stage1_rl_episodes: {stage1_rl_episodes}\n"
        f"stage2_rl_episodes: {stage2_rl_episodes}\n"
        f"search_backend: {blb_v3_search_backend}\n"
        f"resume_run_dir: {resume_run_dir}"
    )
    run_output_dir = str(output_dir or "").strip()
    trainer_output_dir = (
        os.path.join(run_output_dir, "trainer_output")
        if run_output_dir
        else "./inference_output"
    )
    os.makedirs(trainer_output_dir, exist_ok=True)
    seed_everything(final_eval_random_seed)


    device_map = "cuda"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}


    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        **tokenizer_revision_kwargs,
    )
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
        device_map=device_map,
        trust_remote_code=True,
        pad_token_id=tokenizer.pad_token_id,
        **model_revision_kwargs,
    )


    for _param in model.parameters():
        _param.requires_grad_(False)
    model.eval()

    model.to("cuda")


    def tokenize_fn(examples):
        if data_path == "sst2":
            tokenized = tokenizer(
                examples["sentence"],
                truncation=True, padding=False, max_length=128, return_tensors=None,
            )
        else:
            tokenized = tokenizer(
                examples["sentence1"],
                examples["sentence2"],
                truncation=True, padding=False, max_length=128, return_tensors=None,
            )
        return tokenized


    data = load_glue_dataset_equivalent(
        data_path,
        route_log_dir=os.path.join(output_dir, "logs"),
        mrpc_reproducibility_fixture=mrpc_fixture,
    )
    glue_views = resolve_glue_protocol_views(
        data,
        dataset=data_path,
        fixture=glue_fixture,
    )
    if mrpc_fixture is not None:
        mrpc_views = resolve_mrpc_reproducibility_views(
            data,
            data_path=data_path,
            fixture=mrpc_fixture,
        )

    train_data = glue_views.train_full.shuffle(
        seed=final_eval_random_seed
    ).map(tokenize_fn)
    train_probe_data = glue_views.train_probe.map(tokenize_fn)
    if mrpc_views is not None:
        validation_source = mrpc_views.full_validation
    else:
        validation_source = glue_views.validation_full.shuffle(
            seed=final_eval_random_seed
        )
    val_data = validation_source.map(tokenize_fn)

    columns = ["input_ids", "attention_mask", "token_type_ids", "labels"]
    prepared_datasets = []
    for tokenized_data in (train_data, train_probe_data, val_data):
        tokenized_data = tokenized_data.rename_column("label", "labels")
        tokenized_data.set_format(type="torch", columns=columns)
        prepared_datasets.append(tokenized_data)
    train_data, train_probe_data, val_data = prepared_datasets

    glue_protocol_context = GlueDataProtocolContext(
        model_family=model_family,
        dataset=data_path,
        train_probe=train_probe_data,
        validation_full=val_data,
        identity=glue_views.identity,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
        pad_to_multiple_of=8,
    )
    if mrpc_views is not None:
        mrpc_reproducibility = MRPCReproducibilityContext(
            fixture=mrpc_fixture,
            stability_probe=train_probe_data,
        )


    parsed_manual_stage1_gelu = parse_degree_config(manual_stage1_gelu)
    parsed_manual_stage1_softmax = parse_degree_config(manual_stage1_softmax)
    parsed_manual_stage2_noise = parse_noise_config(manual_stage2_noise)
    parsed_stage2_manual_gelu = parse_degree_config(stage2_manual_gelu)
    parsed_stage2_manual_softmax = parse_degree_config(stage2_manual_softmax)
    from layer_importance_evaluator import LayerImportanceEvaluator

    importance_evaluator = LayerImportanceEvaluator(
        model=model,
        train_data=train_data,

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
        stage2_fixed_config_source=stage2_fixed_config_source,
        stage2_fixed_config_path=stage2_fixed_config_path,
        stage2_manual_gelu=parsed_stage2_manual_gelu,
        stage2_manual_softmax=parsed_stage2_manual_softmax,
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
        skip_noise_rl=skip_noise_rl,
        skip_stage1_rl=skip_stage1_rl,
        skip_final_eval=skip_final_eval,
        final_eval_only=final_eval_only,
        resume_run_dir=resume_run_dir,
        decoupled_layout=decoupled_layout,
        stage1_run_id=stage1_run_id,
        data_path=data_path,
        glue_data_protocol=glue_protocol_context,
        mrpc_reproducibility=mrpc_reproducibility,
        stage1_accuracy_tolerance=stage1_accuracy_tolerance,
        stage2_limit_tolerance=stage2_limit_tolerance,
        stage2_stability_tolerance=stage2_stability_tolerance,
        stage2_stability_multiplier=stage2_stability_multiplier,
        stage2_communication_importance_ratio=(
            stage2_communication_importance_ratio
        ),
        stage2_k_trials=stage2_k_trials,
        stage2_probe_size=stage2_probe_size,
        stage2_inference_batch_size=stage2_inference_batch_size,
        blb_v3_inproc_rescale_optimizer_root=(
            blb_v3_inproc_rescale_optimizer_root
            if blb_v3_inproc_rescale_optimizer_root not in (None, "") else None
        ),
        blb_v3_rollout_size=blb_v3_rollout_size,
        blb_v3_eval_interval=blb_v3_eval_interval,
        blb_v3_save_interval=blb_v3_save_interval,
        blb_v3_calibrate_baseline_samples=blb_v3_calibrate_baseline_samples,
        blb_v3_seed=blb_v3_seed,
        blb_v3_reward_devices=blb_v3_reward_devices,
        stage1_rl_devices=stage1_rl_devices,
        blb_v3_online_k_trials=blb_v3_online_k_trials,
        blb_v3_terminal_eval_batch_size=blb_v3_terminal_eval_batch_size,
        blb_v3_promotion_validation_trials=blb_v3_promotion_validation_trials,
        blb_v3_final_selection_top_n=blb_v3_final_selection_top_n,
        blb_v3_final_selection_validation_trials=blb_v3_final_selection_validation_trials,
        blb_v3_promotion_margin_window=blb_v3_promotion_margin_window,
        blb_v3_baseline_groups=blb_v3_baseline_groups,
        blb_v3_baseline_trials_per_group=blb_v3_baseline_trials_per_group,
        blb_v3_constraint_bootstrap_samples=blb_v3_constraint_bootstrap_samples,
        blb_v3_online_constraint_probability=blb_v3_online_constraint_probability,
        blb_v3_promotion_constraint_probability=blb_v3_promotion_constraint_probability,
        blb_v3_final_constraint_probability=blb_v3_final_constraint_probability,
        blb_v3_min_convergence_episodes=blb_v3_min_convergence_episodes,
        blb_v3_convergence_patience_updates=(
            blb_v3_convergence_patience_updates
        ),
        blb_v3_search_backend=blb_v3_search_backend,
        blb_v3_search_evaluation_budget=(
            blb_v3_search_evaluation_budget
        ),
        blb_v3_search_initial_design_size=(
            blb_v3_search_initial_design_size
        ),
        blb_v3_search_candidate_pool_size=(
            blb_v3_search_candidate_pool_size
        ),
        blb_v3_search_population_size=blb_v3_search_population_size,
        blb_v3_search_patience_generations=(
            blb_v3_search_patience_generations
        ),
        blb_v3_search_mutation_max_coordinates=(
            blb_v3_search_mutation_max_coordinates
        ),
        blb_v3_search_rf_n_estimators=blb_v3_search_rf_n_estimators,
        blb_v3_search_rf_min_samples_leaf=(
            blb_v3_search_rf_min_samples_leaf
        ),
        blb_v3_search_full_validation=(
            blb_v3_search_full_validation
        ),
        comparator_smoke=comparator_smoke,
        comparator_stage1_only=comparator_stage1_only,
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            output_dir=trainer_output_dir,
            per_device_eval_batch_size=batch_size,
            disable_tqdm=False,
        ),
        data_collator=data_collator,
        callbacks=[importance_evaluator],
    )

    model.config.use_cache = False
    model.config.is_decoder = False


    print("Starting evaluation...")
    eval_results = trainer.evaluate(eval_dataset=val_data)
    print(f"Final evaluation loss: {eval_results.get('eval_loss')}")


if __name__ == "__main__":
    run_fire_entrypoint(
        fire,
        train,
        program_name="rl_tune.py",
    )
