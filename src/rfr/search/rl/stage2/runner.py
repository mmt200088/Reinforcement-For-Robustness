"""Layerwise Stage-2 orchestration, checkpointing, and strict selection."""
from __future__ import annotations

from collections import deque
import copy
from dataclasses import asdict, dataclass, field
import math
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from rfr.search.runtime.elastic_gpu import (
    ElasticGPUFailure,
    is_recoverable_gpu_failure,
    raise_if_elastic_gpu_restart_requested,
)
from rfr.preparation.data.protocol import PROTOCOL_SCHEMA, TRAIN_PROBE_SPLIT
from rfr.search.common.data_points import (
    RLDataPointWriter,
    make_unique_run_id,
    write_strict_json_file,
)

from rfr.search.common.action_space import K_LEVELS
from rfr.preparation.rescale.baseline_bootstrap import resolve_stage2_model_type
from rfr.search.rl.stage2.policy import (
    BLBStage2LayerwisePolicy,
    LayerwisePolicyConfig,
    LayerwisePPOConfig,
)
from rfr.search.common.truncation_levels import (
    validate_exact_k_domain,
)

if TYPE_CHECKING:
    from rfr.search.common.statistical_constraints import BaselineReference


CUDA_RNG_ROLE_REGISTRY_VERSION = 1
SEARCH_EVIDENCE_SPLIT = TRAIN_PROBE_SPLIT
DATASET_PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
LAYERWISE_RUN_SCHEMA = "stage2_layerwise_train_probe_run_v1"


def merge_cuda_rng_role_registry(
        previous_registry: Optional[Sequence[Any]],
        active_role_states: Sequence[Any],
        ) -> List[Any]:
    """Update active logical roles while retaining temporarily absent roles."""
    registry = list(previous_registry or ())
    for role_index, state in enumerate(active_role_states):
        if role_index < len(registry):
            registry[role_index] = state
        else:
            registry.append(state)
    return registry


def resolve_cuda_rng_role_registry(
        checkpoint: Mapping[str, Any],
        *,
        active_role_count: int,
        new_role_state_factory: Callable[[int], Any],
        ) -> Tuple[List[Any], List[Any]]:
    """Resolve active CUDA RNG streams without tying them to physical GPUs."""
    current_count = int(active_role_count)
    if current_count < 0:
        raise ValueError("active CUDA RNG role count must be non-negative")

    stored_registry = checkpoint.get("cuda_rng_state_by_role")
    if stored_registry is None:
        raise RuntimeError(
            "layerwise checkpoint is missing the CUDA RNG role registry; "
            "a fresh run is required"
        )

    version = int(checkpoint.get("cuda_rng_role_registry_version", 0) or 0)
    if version != CUDA_RNG_ROLE_REGISTRY_VERSION:
        raise RuntimeError(
            "unsupported layerwise checkpoint CUDA RNG role registry "
            f"version: {version}"
        )
    registry = list(stored_registry)
    saved_active_count = int(checkpoint.get(
        "cuda_rng_active_role_count", len(registry),
    ))
    if saved_active_count < 0 or saved_active_count > len(registry):
        raise RuntimeError(
            "layerwise checkpoint CUDA RNG active role count is invalid"
        )
    if current_count == 0 and saved_active_count > 0:
        raise RuntimeError(
            "layerwise checkpoint requires CUDA but no healthy GPU is visible"
        )
    if current_count > 0 and saved_active_count == 0:
        raise RuntimeError(
            "layerwise checkpoint was created without CUDA; "
            "changing the training backend cannot preserve exact results"
        )
    while len(registry) < current_count:
        registry.append(new_role_state_factory(len(registry)))
    return registry, list(registry[:current_count])


def resolve_resumed_best_reward(
        resumed_best: Mapping[str, Any],
        historical_best: Any,
        ) -> float:
    """Return the best finite diagnostic reward available at resume."""
    candidates = (resumed_best.get("reward"), historical_best)
    finite_values: List[float] = []
    for candidate in candidates:
        try:
            value = float(candidate)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value):
            finite_values.append(value)
    return max(finite_values, default=-math.inf)


def _stage2_selection_result_status(summary: Mapping[str, Any]) -> str:
    selected = bool(
        summary.get("best_full_vector") is not None
        and summary.get("best_action_matrix") is not None
        and summary.get("strict_revalidation_status") == "passed"
    )
    return "completed" if selected else "completed_without_selection"


@dataclass
class LayerwiseTrainConfig:
    total_episodes: int = 100
    update_every_n_episodes: int = 4
    log_every_n_episodes: int = 4
    seed: Optional[int] = None
    ppo: LayerwisePPOConfig = field(default_factory=LayerwisePPOConfig)
    absolute_episode_start: int = 0
    planned_total_episodes: Optional[int] = None
    online_num_trials_per_step: int = 3
    terminal_eval_batch_size: int = 4
    promotion_validation_trials: int = 15
    final_selection_validation_trials: int = 15
    baseline_groups: int = 5
    baseline_trials_per_group: int = 3
    online_constraint_probability: float = 0.50
    promotion_constraint_probability: float = 0.80
    final_constraint_probability: float = 0.95
    communication_importance_ratio: float = 1.0


def _seq_log_major_rule(log_fn, title: str, width: int = 68) -> None:
    """Banner: ═══════ on either side of a title line."""
    bar = "═" * int(width)
    log_fn("")
    log_fn(bar)
    log_fn(title if title.startswith(" ") else f"  {title}")
    log_fn(bar)


def _seq_block_title(log_fn, title: str) -> None:
    """Block subtitle: 【…】 brackets."""
    log_fn("")
    log_fn(f"  【{title}】")


def _seq_log_rounded_box(log_fn, lines, indent: str = "  ", min_inner_width: int = 8) -> None:
    """Render multi-line content as a plain, border-less indented block.

    A short separator and bullets remain readable on narrow terminals and with
    mixed-width text.
    """
    stripped = [str(x) for x in lines]
    if not stripped:
        return
    sep = "─" * 4
    log_fn(f"{indent}{sep}")
    for s in stripped:
        log_fn(f"{indent}· {s}")
    log_fn(f"{indent}{sep}")


def _noisy_metric_threshold_from_baseline(
        *,
        noisy_baseline_metric: float,
        tolerance: float,
        ) -> float:
    """Relative metric gate from the noisy baseline.

    ``tolerance`` is a fraction, so ``0.001`` means a strict 0.1% relative drop
    from the noisy all-max BLB baseline. Older code subtracted one probe sample
    (``1 / probe_size``), which made a configured 0.1% gate materially looser
    on MRPC-sized probes.
    """
    baseline = float(noisy_baseline_metric)
    tol = max(0.0, float(tolerance))
    return max(0.0, baseline * (1.0 - tol))


def _noisy_std_threshold_from_baseline(
        *,
        noisy_baseline_std: float,
        stability_multiplier: float,
        floor: float,
        ) -> float:
    """Per-channel stability threshold used by reward.py.

    Stability tolerance is a multiplier on the noisy baseline std. The floor is
    only an absolute minimum for degenerate near-zero baseline variance.
    """
    raw = float(noisy_baseline_std)
    if not np.isfinite(raw):
        raw = 0.0
    return float(max(raw * max(0.0, float(stability_multiplier)), float(floor)))


def _resolve_robust_baseline_config(train_cfg: Any, evaluator: Any) -> Tuple[float, float, int]:
    """Read robust constraint calibration inputs."""
    raw_precision_tolerance = getattr(evaluator, "stage2_limit_tolerance", None)
    precision_tolerance = (
        0.001 if raw_precision_tolerance is None else float(raw_precision_tolerance)
    )
    raw_stability_multiplier = getattr(
        train_cfg,
        "stage2_stability_multiplier",
        getattr(evaluator, "stage2_stability_multiplier", None),
    )
    stability_multiplier = (
        2.0 if raw_stability_multiplier is None else float(raw_stability_multiplier)
    )
    raw_bootstrap_samples = getattr(train_cfg, "constraint_bootstrap_samples", None)
    bootstrap_samples = 4096 if raw_bootstrap_samples is None else int(raw_bootstrap_samples)
    return precision_tolerance, stability_multiplier, bootstrap_samples


def _run_standard_preflight_if_needed(
        *,
        robust_mode: bool,
        run_standard_preflight: Callable[[], None],
        ) -> None:
    """Run the one-shot baseline preflight only outside robust mode."""
    if not robust_mode:
        run_standard_preflight()


def _build_search_gate_env(
        *,
        runner: Any,
        ev: Any,
        base_env: Any,
        train_cfg: Any,
        reward_devices: Sequence[int],
        log: Callable[[str], None],
        ) -> Tuple[Any, int]:
    """Clone the environment while reusing the exact online probe batches."""
    del runner, train_cfg
    dataset_splits = getattr(ev, "dataset_splits", None)
    if not isinstance(dataset_splits, Mapping):
        raise RuntimeError("Stage-2 search gate requires evaluator.dataset_splits")
    train_probe = dataset_splits.get("train_probe")
    if train_probe is None:
        raise RuntimeError("Stage-2 search gate requires train_probe")
    example_count = len(train_probe)

    promotion_env = copy.copy(base_env)
    promotion_env.env_cfg = copy.copy(base_env.env_cfg)
    promotion_env.env_cfg.probe_batch_count = len(base_env.probe_batches)
    promotion_env.env_cfg.persistent_probe_install = False
    promotion_env.probe_batches = base_env.probe_batches
    promotion_env.probe_runner = base_env.probe_runner
    promotion_env.baseline = copy.deepcopy(base_env.baseline)
    promotion_env.reward_weights = copy.deepcopy(base_env.reward_weights)
    promotion_env.statistical_reference = None
    promotion_env.probe_noise_seed = None
    promotion_env._installed_config_fingerprint = None
    promotion_env._installed_action_hash = None
    promotion_env._last_probe_diagnostics = {}

    devices = [int(value) for value in reward_devices]
    if len(devices) >= 2 and promotion_env.probe_runner is None:
        raise RuntimeError(
            "Stage-2 search gate requires the shared train-probe runner"
        )
    log(
        "  * strict search gate: "
        f"split={SEARCH_EVIDENCE_SPLIT} examples={example_count} "
        f"batches={len(base_env.probe_batches)} devices={devices or ['primary']}"
    )
    return promotion_env, int(example_count)


def _install_robust_baseline_reference(
        base_env: Any,
        baseline: Any,
        weights: Any,
        reference: "BaselineReference",
        ) -> None:
    """Install pooled robust constraints into the reward and environment state."""
    baseline.loss_mean = float(reference.loss_mean)
    baseline.metric1_mean = float(reference.metric1_mean)
    baseline.metric2_mean = float(reference.metric2_mean)
    baseline.loss_std = float(reference.loss_std)
    baseline.metric1_std = float(reference.metric1_std)
    baseline.metric2_std = float(reference.metric2_std)
    weights.baseline_metric1 = float(reference.metric1_mean)
    weights.baseline_metric2 = float(reference.metric2_mean)
    weights.stab_tolerance = float(reference.stability_multiplier)
    weights.stab_floor = 0.0
    base_env.statistical_reference = reference
    base_env.loss_threshold = float(reference.loss_limit)
    base_env.acc_threshold = float(reference.metric1_limit)
    base_env.acc_threshold_m2 = float(reference.metric2_limit)
    base_env.stab_threshold = float(reference.loss_std_limit)


def _collect_robust_baseline_reference(
        *,
        base_env: Any,
        baseline_action_vec: Sequence[int],
        base_seed: int,
        precision_tolerance: float,
        stability_multiplier: float,
        bootstrap_samples: int,
        baseline_groups: int = 5,
        trials_per_group: int = 3,
        max_groups: int = 10,
        group_index_start: int = 0,
        ) -> Tuple["BaselineReference", Dict[str, Any]]:
    """Collect deterministic grouped baseline trials for robust constraints."""
    from rfr.search.rl.stage2.seed_utils import derive_baseline_group_probe_seed
    from rfr.search.common.statistical_constraints import (
        DegenerateBaselineVariance,
        TrialSeries,
        build_baseline_reference,
    )

    original_trials = getattr(base_env.env_cfg, "num_trials_per_step", 1)
    original_probe_seed = getattr(base_env, "probe_noise_seed", None)
    reward_weights = getattr(base_env, "reward_weights", None)
    original_reward_design = (
        getattr(reward_weights, "reward_design", None)
        if reward_weights is not None else None
    )
    had_statistical_reference = hasattr(base_env, "statistical_reference")
    original_statistical_reference = getattr(base_env, "statistical_reference", None)
    action = np.asarray(baseline_action_vec, dtype=np.int64).reshape(-1).copy()
    groups: List[Any] = []
    raw_groups: List[Dict[str, Any]] = []

    try:
        if (
                reward_weights is not None
                and str(original_reward_design).strip().lower() == "robust_constrained"
        ):
            reward_weights.reward_design = "stage1_aligned"
        if had_statistical_reference:
            base_env.statistical_reference = None
        required_groups = int(baseline_groups)
        group_trials = int(trials_per_group)
        group_limit = int(max_groups)
        group_start = int(group_index_start)
        if required_groups <= 0 or group_trials <= 0 or group_limit < required_groups:
            raise ValueError("robust baseline group counts must be positive and ordered")
        if group_start < 0:
            raise ValueError("robust baseline group_index_start must be nonnegative")
        if required_groups * group_trials < 15:
            raise ValueError("robust baseline calibration requires at least 15 total trials")
        base_env.env_cfg.num_trials_per_step = group_trials
        for local_group_idx in range(group_limit):
            group_idx = group_start + local_group_idx
            group_probe_seed = derive_baseline_group_probe_seed(base_seed, group_idx)
            base_env.probe_noise_seed = group_probe_seed
            base_env.clear_installed_blb()
            base_env.reset(seed=group_probe_seed)
            _state, _reward, _done, info = base_env.step(action)
            metrics = info.get("metrics") if isinstance(info, Mapping) else None
            if metrics is None:
                raise ValueError("robust baseline probe did not return EpisodeMetrics")

            loss_trials = tuple(float(value) for value in metrics.loss_trials)
            metric1_trials = tuple(float(value) for value in metrics.metric1_trials)
            metric2_trials = tuple(float(value) for value in metrics.metric2_trials)
            trial_seeds = tuple(int(value) for value in metrics.trial_seeds)
            if not (
                    len(loss_trials) == len(metric1_trials) == len(metric2_trials) == group_trials
                    and len(trial_seeds) == group_trials
            ):
                raise ValueError(
                    "robust baseline group raw trials and seeds must match trials_per_group"
                )
            group = TrialSeries(
                loss=loss_trials,
                metric1=metric1_trials,
                metric2=metric2_trials,
                seeds=trial_seeds,
            )
            groups.append(group)
            raw_groups.append({
                "group_index": int(group_idx),
                "group_probe_seed": int(group_probe_seed),
                "trial_seeds": [int(value) for value in trial_seeds],
                "loss_trials": [float(value) for value in loss_trials],
                "metric1_trials": [float(value) for value in metric1_trials],
                "metric2_trials": [float(value) for value in metric2_trials],
            })
            if len(groups) < required_groups:
                continue
            try:
                reference = build_baseline_reference(
                    groups,
                    precision_tolerance=precision_tolerance,
                    stability_multiplier=stability_multiplier,
                    bootstrap_samples=bootstrap_samples,
                    seed=base_seed,
                )
            except DegenerateBaselineVariance as exc:
                if local_group_idx == group_limit - 1:
                    exc.raw_groups = tuple(raw_groups)
                    raise
                continue

            summary = {
                "ok": True,
                "threshold_source": "robust_all_max_blb_baseline",
                "trial_count": int(reference.trial_count),
                "group_count": int(len(groups)),
                "groups": raw_groups,
                "pooled": {
                    "trial_count": int(reference.trial_count),
                    "loss_mean": float(reference.loss_mean),
                    "metric1_mean": float(reference.metric1_mean),
                    "metric2_mean": float(reference.metric2_mean),
                    "loss_std": float(reference.loss_std),
                    "metric1_std": float(reference.metric1_std),
                    "metric2_std": float(reference.metric2_std),
                    "limits": {
                        "loss": float(reference.loss_limit),
                        "metric1": float(reference.metric1_limit),
                        "metric2": float(reference.metric2_limit),
                        "loss_std": float(reference.loss_std_limit),
                        "metric1_std": float(reference.metric1_std_limit),
                        "metric2_std": float(reference.metric2_std_limit),
                    },
                },
                "limits": {
                    "loss": float(reference.loss_limit),
                    "metric1": float(reference.metric1_limit),
                    "metric2": float(reference.metric2_limit),
                    "loss_std": float(reference.loss_std_limit),
                    "metric1_std": float(reference.metric1_std_limit),
                    "metric2_std": float(reference.metric2_std_limit),
                },
                "bootstrap": {
                    "samples": int(reference.bootstrap_samples),
                    "seed": int(reference.bootstrap_seed),
                },
                "precision_tolerance": float(reference.precision_tolerance),
                "stability_multiplier": float(reference.stability_multiplier),
            }
            return reference, summary
    finally:
        try:
            base_env.clear_installed_blb()
        finally:
            base_env.env_cfg.num_trials_per_step = original_trials
            base_env.probe_noise_seed = original_probe_seed
            if reward_weights is not None and original_reward_design is not None:
                reward_weights.reward_design = original_reward_design
            if had_statistical_reference:
                base_env.statistical_reference = original_statistical_reference

    raise AssertionError("robust baseline collection exhausted without a result")


def _build_layerwise_candidate_identity_context(
        *,
        train_cfg: Any,
        evaluator: Any,
        fusion_map: Any,
        max_sfs: Any,
        fixed_gelu: np.ndarray,
        fixed_softmax: np.ndarray,
        robust_reference: Any,
        authoritative_robust_reference: Any,
        validation_banks: Any,
        probe_example_count: int,
        authoritative_example_count: int,
        schedule: Sequence[Any],
        static_skeletons_baseline: Any,
        algorithm_contract: Mapping[str, Any],
        algorithm_contract_hash: str,
        stage1_selection_binding: Mapping[str, Any] | None = None,
        ) -> Dict[str, Any]:
    """Bind layerwise evidence to the ordinary scientific run context."""
    from rfr.search.common.candidate_store import build_candidate_identity_context, sha256_json
    from rfr.preparation.data.protocol import (
        PROTOCOL_SCHEMA as DATASET_PROTOCOL_SCHEMA,
        TRAIN_PROBE_SPLIT as SEARCH_EVIDENCE_SPLIT,
    )
    from rfr.search.common.layerwise_action import (
        K_LEVELS,
        LAYERWISE_COST_MODEL_REVISION,
        LAYERWISE_DECODE_VERSION,
        layerwise_action_space_version,
    )
    from rfr.search.rl.stage2.layerwise_runner import bind_layerwise_candidate_identity

    stage1_degrees = {
        "gelu": [int(value) for value in fixed_gelu.reshape(-1)],
        "softmax": [int(value) for value in fixed_softmax.reshape(-1)],
    }
    stage1_binding_payload = None
    if isinstance(stage1_selection_binding, Mapping):
        stage1_binding_payload = {
            "schema_version": str(
                stage1_selection_binding.get("schema_version") or ""
            ),
            "algorithm": str(
                stage1_selection_binding.get("algorithm") or ""
            ),
            "model_type": str(
                stage1_selection_binding.get("model_type") or ""
            ),
            "dataset": str(stage1_selection_binding.get("dataset") or ""),
            "gelu_degrees": [
                int(value)
                for value in stage1_selection_binding.get(
                    "gelu_degrees", ()
                )
            ],
            "softmax_degrees": [
                int(value)
                for value in stage1_selection_binding.get(
                    "softmax_degrees", ()
                )
            ],
            "num_layers": int(stage1_selection_binding.get("num_layers", 0)),
            "config_path": os.path.abspath(os.fspath(
                stage1_selection_binding.get("config_path") or ""
            )),
            "config_sha256": str(
                stage1_selection_binding.get("config_sha256") or ""
            ),
            "dataset_protocol_hash": str(
                stage1_selection_binding.get("dataset_protocol_hash") or ""
            ),
        }
        config_sha256 = stage1_binding_payload["config_sha256"]
        if (
                len(config_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in config_sha256
                )
        ):
            raise RuntimeError(
                "strict candidate identity requires a valid Stage-1 JSON hash"
            )
        if (
                stage1_binding_payload["gelu_degrees"]
                != stage1_degrees["gelu"]
                or stage1_binding_payload["softmax_degrees"]
                != stage1_degrees["softmax"]
                or stage1_binding_payload["num_layers"]
                != len(stage1_degrees["gelu"])
                or stage1_binding_payload["dataset_protocol_hash"]
                != str(getattr(evaluator, "dataset_protocol_hash", "") or "")
                or stage1_binding_payload["config_path"]
                != str(
                    getattr(evaluator, "stage1_best_config_input_path", "")
                    or ""
                )
                or config_sha256
                != str(
                    getattr(evaluator, "stage1_best_config_input_sha256", "")
                    or ""
                )
        ):
            raise RuntimeError(
                "strict candidate identity does not match Stage-1 binding"
            )
    num_layers = len(stage1_degrees["gelu"])
    stage2_inference_batch_size = int(
        getattr(train_cfg, "stage2_inference_batch_size", None)
        or getattr(evaluator, "stage2_inference_batch_size", None)
        or getattr(evaluator, "batch_size", 1)
    )
    comparator_batch_identity = (
        {"stage2_inference_batch_size": stage2_inference_batch_size}
        if stage1_binding_payload is not None else {}
    )
    dataset_protocol_hash = str(
        getattr(evaluator, "dataset_protocol_hash", "") or ""
    )

    def reference_payload(reference: Any) -> Dict[str, Any]:
        return {
            "precision_tolerance": float(reference.precision_tolerance),
            "stability_multiplier": float(reference.stability_multiplier),
            "bootstrap_seed": int(reference.bootstrap_seed),
            "bootstrap_samples": int(reference.bootstrap_samples),
            "limits": {
                "loss": float(reference.loss_limit),
                "metric1": float(reference.metric1_limit),
                "metric2": float(reference.metric2_limit),
                "loss_std": float(reference.loss_std_limit),
                "metric1_std": float(reference.metric1_std_limit),
                "metric2_std": float(reference.metric2_std_limit),
            },
        }

    threshold_policy = {
        **comparator_batch_identity,
        "dataset_protocol_schema": DATASET_PROTOCOL_SCHEMA,
        "dataset_protocol_hash": dataset_protocol_hash,
        "search_split": SEARCH_EVIDENCE_SPLIT,
        "precision_tolerance": float(robust_reference.precision_tolerance),
        "stability_multiplier": float(robust_reference.stability_multiplier),
        "bootstrap_seed": int(robust_reference.bootstrap_seed),
        "bootstrap_samples": int(robust_reference.bootstrap_samples),
        "online_constraint_probability": float(
            getattr(train_cfg, "online_constraint_probability", 0.50)
        ),
        "promotion_constraint_probability": float(
            getattr(train_cfg, "promotion_constraint_probability", 0.80)
        ),
        "final_constraint_probability": float(
            getattr(train_cfg, "final_constraint_probability", 0.95)
        ),
        "limits": {
            "loss": float(robust_reference.loss_limit),
            "metric1": float(robust_reference.metric1_limit),
            "metric2": float(robust_reference.metric2_limit),
            "loss_std": float(robust_reference.loss_std_limit),
            "metric1_std": float(robust_reference.metric1_std_limit),
            "metric2_std": float(robust_reference.metric2_std_limit),
        },
        "evidence_tiers": {
            "F1": {
                "split": SEARCH_EVIDENCE_SPLIT,
                "example_count": int(probe_example_count),
                "reference": reference_payload(robust_reference),
            },
            "F4": {
                "split": SEARCH_EVIDENCE_SPLIT,
                "example_count": int(authoritative_example_count),
                "reference": reference_payload(authoritative_robust_reference),
                "validation_banks": validation_banks.contract_payload(),
            },
        },
    }
    rescale_root = os.path.realpath(str(train_cfg.inproc_rescale_optimizer_root))
    model_type = resolve_stage2_model_type(
        str(getattr(evaluator, "model_type", "") or ""),
        num_layers=num_layers,
    )
    context = build_candidate_identity_context(
        action_space_version=layerwise_action_space_version(num_layers),
        registry_hash=sha256_json(fusion_map),
        max_sfs_hash=sha256_json(max_sfs),
        stage1_config_content_hash=sha256_json(stage1_degrees),
        stage1_gelu_degrees=stage1_degrees["gelu"],
        stage1_softmax_degrees=stage1_degrees["softmax"],
        profile=str(train_cfg.profile),
        rescale_optimizer_mode="in_process_real",
        rescale_optimizer_root=rescale_root,
        rescale_optimizer_canonical_hash=sha256_json({
            "root": rescale_root,
            "static_skeletons": static_skeletons_baseline,
        }),
        decode_version=LAYERWISE_DECODE_VERSION,
        dataset=str(train_cfg.profile),
        model=model_type,
        metric_policy_version="robust_bootstrap_5x3_v1",
        threshold_policy_hash=sha256_json(threshold_policy),
        mask_schedule_hash=sha256_json(schedule),
    )
    return bind_layerwise_candidate_identity(
        context,
        K_LEVELS,
        LAYERWISE_COST_MODEL_REVISION,
        {
            "algorithm_contract_hash": str(algorithm_contract_hash),
            "dataset_protocol_schema": DATASET_PROTOCOL_SCHEMA,
            "dataset_protocol_hash": dataset_protocol_hash,
            "search_split": SEARCH_EVIDENCE_SPLIT,
            **comparator_batch_identity,
            **(
                {"stage1_selection_binding": stage1_binding_payload}
                if stage1_binding_payload is not None else {}
            ),
            "communication_importance_ratio": algorithm_contract[
                "communication_importance_ratio"
            ],
            "compute_axis_denominator": algorithm_contract[
                "compute_axis_denominator"
            ],
            "communication_axis_denominator": algorithm_contract[
                "communication_axis_denominator"
            ],
            "resource_credit_mode": algorithm_contract[
                "resource_credit_mode"
            ],
            "strict_resource_order": algorithm_contract[
                "strict_resource_order"
            ],
        },
    )


def _run_layerwise_training_branch(
        *,
        train_cfg: Any,
        evaluator: Any,
        base_env: Any,
        fusion_map: Any,
        max_sfs: Any,
        robust_reference: Any,
        promotion_base_env: Any,
        authoritative_robust_reference: Any,
        authoritative_robust_summary: Optional[Mapping[str, Any]],
        authoritative_validation_banks: Any,
        authoritative_validation_example_count: int,
        static_skeletons_baseline: Any,
        baseline_action_vec: Sequence[int],
        fixed_gelu: np.ndarray,
        fixed_softmax: np.ndarray,
        fixed_label: str,
        fixed_source: str,
        blb_progress_dir: str,
        clean_baseline_metrics: Any,
        baseline_preflight_metrics: Mapping[str, Any],
        status: Any,
        resume_checkpoint_path: Any,
        run_lock: Any,
        log: Callable[[str], None],
        ) -> Dict[str, Any]:
    """Run the production layerwise PPO pipeline."""
    if robust_reference is None:
        raise RuntimeError("layerwise robust PPO requires a calibrated statistical reference")
    if (
            promotion_base_env is None
            or authoritative_robust_reference is None
            or authoritative_validation_banks is None
    ):
        raise RuntimeError(
            "layerwise robust PPO requires the strict train-probe evaluator"
        )
    bullet = "*"

    from rfr.common.json_utils import to_jsonable

    from rfr.search.common.candidate_store import CandidateStore, sha256_json
    from rfr.search.common.diagnostics import EpisodeStats, PPOUpdateStats, RLDiagnosticsRecorder
    from rfr.preparation.fusion.fixed_action import build_fusion_fixed_config
    from rfr.search.common.layerwise_action import (
        K_LEVELS as LAYERWISE_K_LEVELS,
        LAYERWISE_COST_MODEL_REVISION,
        LAYERWISE_DECODE_VERSION,
        LAYERWISE_SLOT_NAMES,
        describe_layerwise_action_matrix,
        layerwise_action_space_version,
        max_communication_saving_units,
        max_compute_saving_units,
    )
    from rfr.search.rl.stage2.layerwise_env import BLBStage2LayerwiseEnv
    from rfr.search.rl.stage2.layerwise_runner import (
        _PROBABILITY_FIELDS,
        CheckpointFileFingerprintTracker,
        StrictSelectionKey,
        _to_plain_mapping,
        build_layerwise_run_context,
        initialize_layerwise_policy,
        normalized_constraint_safety_margins,
        resolve_layerwise_episode_budget,
        strict_selection_key,
        strict_selection_key_from_snapshot,
        train_layerwise,
        validate_fresh_layerwise_run_state,
        validate_layerwise_checkpoint_metadata,
        validate_layerwise_episode_limit_extension,
    )
    from rfr.search.rl.stage2.policy_network import (
        POLICY_NETWORK_ID,
        POLICY_RL_VARIANT,
        bind_policy_network_contract,
        policy_network_architecture,
        validate_checkpoint_policy_network,
    )
    from rfr.search.common.persistence import write_training_curves
    from rfr.search.common.precision_presets import (
        PRECISION_PRESET_VERSION,
        PRECISION_PRESETS,
        allocated_precision_tolerances,
        network_axis_weights,
        validate_communication_importance_ratio,
    )
    layerwise_manifest_path = os.path.join(
        blb_progress_dir, "layerwise_run_manifest.json",
    )
    requested_total_episodes = int(train_cfg.total_episodes)
    resolve_layerwise_episode_budget(requested_total_episodes, 0)
    algorithm_revision = "network_weighted_hml_max_episodes_v13"
    policy_network_id = POLICY_NETWORK_ID
    policy_architecture = policy_network_architecture()
    rl_variant = POLICY_RL_VARIANT
    layerwise_entropy_regularization = {
        "kind": "disabled",
        "coefficient": 0.0,
        "optimization_role": "monitor_only",
    }
    layerwise_termination = {
        "mode": "maximum_episodes",
        "episode_limit": requested_total_episodes,
        "strict_certification_at_episode_limit": True,
        "strict_revalidation_trials": int(
            authoritative_validation_banks.bank_c.trial_count
        ),
        "strict_revalidation_diagnostic_probability": float(
            getattr(train_cfg, "final_constraint_probability", 0.95)
        ),
        "selection_order": (
            "feasible,weighted_resource_score,balance_tiebreak,confidence_vector,"
            "safety_margin_vector,"
            "action_lexicographic"
        ),
        "entropy_role": "diagnostic_only",
        "validation_banks": authoritative_validation_banks.contract_payload(),
        "counts_only_finite_ppo_updates": True,
    }
    algorithm_termination = dict(layerwise_termination)
    algorithm_termination["episode_limit"] = "runtime_extendable"
    layerwise_ppo_mode = {
        "factorized_actor_clip": True,
        "behavior_log_prob_source": "sampling_time_per_slot_v1",
        "actor_credit_mode": "shared_constraint_plus_separable_axis_resource",
        "actor_advantage_normalization": "per_slot_center_shared_scale_v1",
        "entropy_average_active_slots": True,
        "entropy_normalize_active_slots": True,
    }
    layerwise_env = BLBStage2LayerwiseEnv(
        base_env=base_env,
        fusion_map=fusion_map,
        baseline_action_vec=baseline_action_vec,
        profile=str(train_cfg.profile),
        communication_importance_ratio=(
            validate_communication_importance_ratio(
                getattr(train_cfg, "communication_importance_ratio", 1.0),
            )
        ),
    )
    layerwise_horizon = int(layerwise_env.horizon)
    if layerwise_horizon != int(evaluator.total_layers):
        raise RuntimeError(
            "layerwise environment/model depth mismatch: "
            f"{layerwise_horizon} != {int(evaluator.total_layers)}"
        )
    online_probe_example_count = sum(
        int(batch.labels.numel()) for batch in base_env.probe_batches
    )
    if online_probe_example_count <= 0:
        raise RuntimeError("layerwise F1 probe must contain at least one example")
    if online_probe_example_count != 256:
        raise RuntimeError(
            "layerwise F1 probe must contain exactly 256 stratified examples; "
            f"received {online_probe_example_count}"
        )
    from rfr.search.comparators.common.stage2_core import normalize_search_backend

    search_backend = normalize_search_backend(
        getattr(train_cfg, "search_backend", "ppo")
    )
    if search_backend != "ppo":
        expected_stage1_source = (
            f"stage1_json:{getattr(evaluator, 'stage1_best_config_input_path', '')}"
        )
        if str(fixed_source) != expected_stage1_source:
            raise RuntimeError(
                "two-stage comparator must bind its own Stage-1 result before "
                f"Stage-2: expected fixed_source={expected_stage1_source!r}, "
                f"got {str(fixed_source)!r}"
            )
        from rfr.search.comparators.common.stage2_runner import (
            canonical_strict_validation,
            run_layerwise_search_baseline,
        )

        search_output_dir = os.path.join(
            blb_progress_dir, f"search_{search_backend}",
        )
        from rfr.common.json_utils import read_json_file

        invocation_path = os.path.join(
            search_output_dir, "invocation.json",
        )
        if not os.path.isfile(invocation_path):
            raise RuntimeError("Stage-2 search has no invocation.json")
        invocation_contract = read_json_file(invocation_path)
        expected_invocation = _build_search_invocation_contract(
            runner=SimpleNamespace(evaluator=evaluator),
            train_cfg=train_cfg,
            fixed_gelu=fixed_gelu,
            fixed_softmax=fixed_softmax,
            fixed_label=fixed_label,
            fixed_source=fixed_source,
        )
        if (
                not isinstance(invocation_contract, Mapping)
                or dict(invocation_contract) != expected_invocation
        ):
            raise RuntimeError(
                "Stage-2 search invocation does not match Stage-1 binding"
            )
        stage1_selection_binding = dict(
            expected_invocation["stage1_selection_binding"]
        )
        communication_ratio = float(
            layerwise_env.communication_importance_ratio
        )
        stage2_inference_batch_size = int(
            getattr(train_cfg, "stage2_inference_batch_size")
        )
        backend_contract = {
            "bo_rf": {
                "proposal": "categorical_rf_pof_times_deterministic_ei_v1",
                "duplicate_policy": "unique_action_cache_v1",
            },
            "greedy": {
                "proposal": "exhaustive_1opt_then_2opt_return_to_1opt_v1",
                "duplicate_policy": "unique_action_cache_v1",
            },
            "coinn_ga": {
                "proposal": "fitness_weighted_adjacent_mutation_no_crossover_v1",
                "duplicate_policy": "unique_action_cache_v1",
                "collision_fallback": (
                    "deterministic_complete_adjacent_neighborhood_v1"
                ),
            },
        }[search_backend]
        search_config = {
            "initial_design_size": int(
                train_cfg.search_initial_design_size
            ),
            "candidate_pool_size": int(
                train_cfg.search_candidate_pool_size
            ),
            "population_size": int(
                train_cfg.search_population_size
            ),
            "bo_no_improvement_patience": int(
                train_cfg.search_bo_no_improvement_patience
            ),
            "greedy_no_improvement_rounds": int(
                train_cfg.search_greedy_no_improvement_rounds
            ),
            "ga_generations": int(
                train_cfg.search_ga_generations
            ),
            "mutation_max_coordinates": int(
                train_cfg.search_mutation_max_coordinates
            ),
            "rf_n_estimators": int(
                train_cfg.search_rf_n_estimators
            ),
            "rf_min_samples_leaf": int(
                train_cfg.search_rf_min_samples_leaf
            ),
        }
        search_contract = {
            "schema_version": "stage2_search_baseline_contract_v3",
            "dataset_protocol_schema": DATASET_PROTOCOL_SCHEMA,
            "dataset_protocol_hash": getattr(
                evaluator, "dataset_protocol_hash", None
            ),
            "search_split": SEARCH_EVIDENCE_SPLIT,
            "search_backend": search_backend,
            "backend_contract": backend_contract,
            "stage2_inference_batch_size": stage2_inference_batch_size,
            "action_space_version": layerwise_action_space_version(
                layerwise_horizon
            ),
            "decode_version": LAYERWISE_DECODE_VERSION,
            "cost_model_revision": LAYERWISE_COST_MODEL_REVISION,
            "communication_importance_ratio": communication_ratio,
            "compute_axis_denominator": int(
                max_compute_saving_units(layerwise_horizon)
            ),
            "communication_axis_denominator": int(
                max_communication_saving_units(layerwise_horizon)
            ),
            "resource_credit_mode": "separable_weighted_per_slot_v1",
            "strict_resource_order": [
                "weighted_score", "balance_tiebreak",
            ],
            "online_gate": (
                "six_bootstrap_probabilities_at_online_threshold"
            ),
            "online_gate_margin_basis": (
                "constraint_probability_minus_online_threshold_v1"
            ),
            "online_seed_contract": "ppo_global_evaluation_index_v1",
            "candidate_repeat_policy": "unique_action_cache_per_optimizer_v1",
            "strict_gate": (
                "joint_six_point_plus_compute_and_communication_"
                "counterfactual_six_point_v1"
            ),
            "bootstrap_probability_role_at_f4": (
                "diagnostic_tiebreak_only"
            ),
            "strict_selection_tiebreak": (
                "full_materialized_vector_then_f4_candidate_key_v1"
            ),
            "validation_banks": (
                authoritative_validation_banks.contract_payload()
            ),
            "online_constraint_probability": float(
                getattr(train_cfg, "online_constraint_probability", 0.50)
            ),
            "promotion_constraint_probability": float(
                getattr(train_cfg, "promotion_constraint_probability", 0.80)
            ),
            "final_constraint_probability": float(
                getattr(train_cfg, "final_constraint_probability", 0.95)
            ),
            "search_config": search_config,
        }
        search_contract_hash = sha256_json(search_contract)

        search_manifest = {
            "dataset_protocol_schema": DATASET_PROTOCOL_SCHEMA,
            "dataset_protocol_hash": getattr(
                evaluator, "dataset_protocol_hash", None
            ),
            "search_split": SEARCH_EVIDENCE_SPLIT,
            "profile": str(train_cfg.profile),
            "model_type": str(getattr(evaluator, "model_type", "") or ""),
            "num_layers": int(layerwise_horizon),
            "fixed_gelu": [
                int(value) for value in np.asarray(fixed_gelu).reshape(-1)
            ],
            "fixed_softmax": [
                int(value) for value in np.asarray(fixed_softmax).reshape(-1)
            ],
            "fixed_label": str(fixed_label),
            "fixed_source": str(fixed_source),
            "stage2_invocation": invocation_contract,
            "stage2_inference_batch_size": stage2_inference_batch_size,
            "stage1_backend": search_backend,
            "stage1_bound_into_stage2": bool(
                str(fixed_source).startswith("stage1_json:")
            ),
            "stage1_selection_binding": stage1_selection_binding,
            "online_fidelity": {
                "split": SEARCH_EVIDENCE_SPLIT,
                "example_count": int(online_probe_example_count),
                "batch_size": stage2_inference_batch_size,
                "trials_per_action": int(
                    base_env.env_cfg.num_trials_per_step
                ),
            },
            "authoritative_fidelity": {
                "split": SEARCH_EVIDENCE_SPLIT,
                "example_count": int(
                    authoritative_validation_example_count
                ),
                "batch_size": stage2_inference_batch_size,
                "validation_banks": (
                    authoritative_validation_banks.contract_payload()
                ),
            },
            "constraint_limits": {
                "loss": float(robust_reference.loss_limit),
                "metric1": float(robust_reference.metric1_limit),
                "metric2": float(robust_reference.metric2_limit),
                "loss_std": float(robust_reference.loss_std_limit),
                "metric1_std": float(robust_reference.metric1_std_limit),
                "metric2_std": float(robust_reference.metric2_std_limit),
            },
            "scientific_status": "full_search_with_strict_train_probe_gate",
            "algorithm_contract": search_contract,
            "algorithm_contract_hash": search_contract_hash,
        }
        strict_candidate_store = CandidateStore(os.path.join(
            search_output_dir, "strict_candidate_store.jsonl",
        ))
        strict_identity_context = (
            _build_layerwise_candidate_identity_context(
                train_cfg=train_cfg,
                evaluator=evaluator,
                fusion_map=fusion_map,
                max_sfs=max_sfs,
                fixed_gelu=fixed_gelu,
                fixed_softmax=fixed_softmax,
                robust_reference=robust_reference,
                authoritative_robust_reference=(
                    authoritative_robust_reference
                ),
                validation_banks=authoritative_validation_banks,
                probe_example_count=int(online_probe_example_count),
                authoritative_example_count=int(
                    authoritative_validation_example_count
                ),
                schedule=layerwise_env.schedule,
                static_skeletons_baseline=(
                    static_skeletons_baseline
                ),
                algorithm_contract=search_contract,
                algorithm_contract_hash=search_contract_hash,
                stage1_selection_binding=stage1_selection_binding,
            )
        )
        search_manifest["strict_identity_context_hash"] = sha256_json(
            strict_identity_context
        )
        search_manifest["strict_candidate_store"] = os.fspath(
            strict_candidate_store.path
        )
        def strict_validator(search_result):
            return canonical_strict_validation(
                result=search_result,
                layerwise_env=layerwise_env,
                promotion_base_env=promotion_base_env,
                candidate_store=strict_candidate_store,
                identity_context=strict_identity_context,
                validation_banks=authoritative_validation_banks,
                top_n=int(train_cfg.final_selection_top_n),
                communication_importance_ratio=communication_ratio,
                promotion_probability=float(getattr(
                    train_cfg,
                    "promotion_constraint_probability",
                    0.80,
                )),
                final_probability=float(getattr(
                    train_cfg,
                    "final_constraint_probability",
                    0.95,
                )),
            )

        from rfr.search.common.statistical_constraints import (
            baseline_reference_resume_payload,
        )

        def pending_strict_context_writer(resume_contract):
            _write_pending_strict_resume_context(
                search_output_dir=search_output_dir,
                invocation_contract=invocation_contract,
                resume_contract=resume_contract,
                clean_baseline_metrics=(
                    _episode_metrics_resume_payload(clean_baseline_metrics)
                ),
                robust_reference=baseline_reference_resume_payload(
                    robust_reference
                ),
                baseline_preflight_metrics=baseline_preflight_metrics,
                validation_banks=authoritative_validation_banks.resume_payload(),
                authoritative_robust_summary=authoritative_robust_summary,
                authoritative_validation_example_count=(
                    authoritative_validation_example_count
                ),
            )

        run_lock.bind_context(sha256_json({
            **search_manifest,
            "search_backend": search_backend,
            "seed": int(train_cfg.seed),
        }))
        status.set_phase(f"Stage-2 {search_backend} search")
        search_run = run_layerwise_search_baseline(
            backend=search_backend,
            layerwise_env=layerwise_env,
            robust_reference=robust_reference,
            output_dir=search_output_dir,
            seed=int(train_cfg.seed),
            initial_design_size=int(train_cfg.search_initial_design_size),
            candidate_pool_size=int(train_cfg.search_candidate_pool_size),
            population_size=int(train_cfg.search_population_size),
            bo_no_improvement_patience=int(
                train_cfg.search_bo_no_improvement_patience
            ),
            greedy_no_improvement_rounds=int(
                train_cfg.search_greedy_no_improvement_rounds
            ),
            ga_generations=int(train_cfg.search_ga_generations),
            mutation_max_coordinates=int(
                train_cfg.search_mutation_max_coordinates
            ),
            rf_n_estimators=int(train_cfg.search_rf_n_estimators),
            rf_min_samples_leaf=int(train_cfg.search_rf_min_samples_leaf),
            communication_importance_ratio=float(
                layerwise_env.communication_importance_ratio
            ),
            manifest=search_manifest,
            strict_validator=strict_validator,
            pending_strict_context_writer=(
                pending_strict_context_writer
            ),
        )
        selected = search_run["selected"]
        selected_metadata = dict(selected.metadata)
        best_full_vector = [
            int(value)
            for value in selected_metadata.get("pending_full_vector", [])
        ]
        best_action_matrix = [
            [int(value) for value in row]
            for row in selected.action_matrix
        ]
        selected_action_identity = _selected_action_identity_payload(
            selected
        )
        if not best_full_vector:
            raise RuntimeError(
                "search baseline selected action has no materialized full vector"
            )
        best_layer_configurations = describe_layerwise_action_matrix(
            best_action_matrix
        )
        limits = selected.limits.as_dict()
        status.update_after_episode(
            int(search_run["result"].evaluation_count),
            float(selected.reward or 0.0),
            {
                "priority": 3 if selected.feasible else 1,
                "invalid": not bool(selected.valid),
                "search_backend": search_backend,
            },
        )
        fixed_config = build_fusion_fixed_config(
            best_full_vector,
            profile=str(train_cfg.profile),
            num_layers=int(evaluator.total_layers),
            gelu=np.asarray(fixed_gelu, dtype=int),
            softmax=np.asarray(fixed_softmax, dtype=int),
            fusion_map=fusion_map,
            source=f"stage2_{search_backend}_best",
        )
        best_action_group = dict(fixed_config["group"])
        best_action_group["policy_actions"] = best_action_matrix
        best_action_group["boosted_overrides"] = selected_metadata.get(
            "boosted_overrides", []
        )
        status.set_best(
            float(selected.reward or 0.0),
            best_action_vec=best_full_vector,
            best_breakdown=selected.as_dict(),
            best_episode=int(search_run["result"].evaluation_count),
        )
        strict_feasible = bool(search_run.get("strict_feasible", False))
        completion_status = (
            "completed" if strict_feasible else "completed_infeasible"
        )
        scientific_status = (
            "full_search_with_strict_train_probe_gate"
            if strict_feasible
            else "full_search_strict_least_violating"
        )
        status.set_phase(f"Stage-2 {search_backend} search complete")
        return {
            "dataset_protocol_hash": getattr(
                evaluator, "dataset_protocol_hash", None
            ),
            "fixed_gelu": np.asarray(fixed_gelu, dtype=int).copy(),
            "fixed_softmax": np.asarray(fixed_softmax, dtype=int).copy(),
            "status": completion_status,
            "scientific_status": scientific_status,
            "strict_feasible": strict_feasible,
            "stage2_inference_batch_size": stage2_inference_batch_size,
            "selected_action_identity": selected_action_identity,
            "search_backend": search_backend,
            "stage1_consumed_binding": stage1_selection_binding,
            "strict_identity_context_hash": search_manifest[
                "strict_identity_context_hash"
            ],
            "final_config_fingerprint": selected_action_identity[
                "final_config_fingerprint"
            ],
            "search_accounting": {
                key: search_run["manifest"].get(key)
                for key in (
                    "seed",
                    "observation_count",
                    "inference_reaching_candidate_count",
                    "online_candidate_trial_count",
                    "strict_evaluated_candidate_count",
                    "strict_joint_trial_count",
                    "strict_compute_trial_count",
                    "strict_communication_trial_count",
                    "strict_total_evidence_trial_count",
                    "strict_fresh_trial_count",
                    "total_candidate_trial_count",
                    "model_forward_trial_count",
                    "online_search_wall_seconds",
                    "strict_attempt_count",
                    "strict_attempt_wall_seconds_total",
                    "strict_validation_wall_seconds",
                    "total_wall_seconds",
                    "termination_reason",
                )
            },
            "rl_variant": f"blb_v3_layerwise_search_{search_backend}",
            "blb_v3_best_action_vec": best_full_vector,
            "blb_v3_best_action_group": best_action_group,
            "blb_v3_layerwise_best_action_group": best_action_group,
            "blb_v3_layerwise_best_configuration": (
                best_layer_configurations
            ),
            "blb_v3_best_reward": float(selected.reward or 0.0),
            "blb_v3_profile": str(train_cfg.profile),
            "blb_v3_fusion_count_action": True,
            "blb_v3_total_episodes": int(
                search_run["result"].evaluation_count
            ),
            "limit_loss": float(limits["loss_max"]),
            "limit_p": float(limits["metric1_min"]),
            "limit_s": float(limits["metric2_min"]),
            "proxy_limit_loss": float(limits["loss_max"]),
            "proxy_limit_p": float(limits["metric1_min"]),
            "proxy_limit_s": float(limits["metric2_min"]),
            "search_limits": {
                "loss": float(limits["loss_max"]),
                "metric1": float(limits["metric1_min"]),
                "metric2": float(limits["metric2_min"]),
                "loss_std": float(limits["loss_std_max"]),
                "metric1_std": float(limits["metric1_std_max"]),
                "metric2_std": float(limits["metric2_std_max"]),
            },
            "selection_diagnostics": {
                "selection_mode": (
                    "layerwise_constrained_search_baseline"
                ),
                "best_action_matrix": best_action_matrix,
                "best_layer_configurations": best_layer_configurations,
                "best_evaluation": selected.as_dict(),
                "strict_validation": search_run["strict_validation"],
                "artifact_paths": search_run["artifact_paths"],
            },
            "layerwise_summary": {
                **search_run["result"].as_dict(),
                "selected": selected.as_dict(),
                "strict_validation": search_run["strict_validation"],
                "artifact_paths": search_run["artifact_paths"],
            },
        }
    policy_cfg = LayerwisePolicyConfig(
        state_dim=int(layerwise_env.state_dim),
        max_step_dim=len(LAYERWISE_SLOT_NAMES),
        max_num_levels=max(2, len(PRECISION_PRESETS)),
        horizon=layerwise_horizon,
        num_layers=layerwise_horizon,
        **policy_architecture,
    )
    ppo = LayerwisePPOConfig(
        lr=float(train_cfg.ppo.lr),
        clip_range=float(train_cfg.ppo.clip_range),
        n_epochs=int(train_cfg.ppo.n_epochs),
        minibatch_size=int(train_cfg.ppo.minibatch_size),
        ent_coef=0.0,
        value_coef=float(train_cfg.ppo.value_coef),
        max_grad_norm=float(train_cfg.ppo.max_grad_norm),
        gamma=1.0,
        gae_lambda=1.0,
        per_slot_entropy_recovery=False,
        factorized_actor_clip=True,
        entropy_average_active_slots=True,
        entropy_normalize_active_slots=True,
    )
    algorithm_contract = {
        "schema_version": "stage2_layerwise_algorithm_contract_v7",
        "dataset_protocol_schema": DATASET_PROTOCOL_SCHEMA,
        "dataset_protocol_hash": getattr(
            evaluator, "dataset_protocol_hash", None
        ),
        "search_split": SEARCH_EVIDENCE_SPLIT,
        "algorithm_revision": algorithm_revision,
        "rl_variant": rl_variant,
        "action_space_version": layerwise_action_space_version(
            layerwise_horizon
        ),
        "decode_version": LAYERWISE_DECODE_VERSION,
        "cost_model_revision": LAYERWISE_COST_MODEL_REVISION,
        "k_levels": [int(value) for value in LAYERWISE_K_LEVELS],
        "precision_preset_version": PRECISION_PRESET_VERSION,
        "precision_presets": [
            {
                "name": preset.name,
                "k_by_block": list(preset.simulation_k_by_block),
                "communication_utility": float(preset.communication_utility),
            }
            for preset in PRECISION_PRESETS
        ],
        "communication_importance_ratio": float(
            layerwise_env.communication_importance_ratio
        ),
        "network_axis_weights": list(network_axis_weights(
            layerwise_env.communication_importance_ratio,
        )),
        "axis_precision_tolerances": list(allocated_precision_tolerances(
            float(robust_reference.precision_tolerance),
            layerwise_env.communication_importance_ratio,
        )),
        "compute_axis_denominator": int(
            max_compute_saving_units(layerwise_horizon)
        ),
        "communication_axis_denominator": int(
            max_communication_saving_units(layerwise_horizon)
        ),
        "resource_credit_mode": "separable_weighted_per_slot_v1",
        "strict_resource_order": ["weighted_score", "balance_tiebreak"],
        "resource_objective": {
            "compute_axis": "learnable_block4_fusion_count",
            "communication_axis": "layerwise_precision_preset_utility",
            "selection": "network_weighted_sum_then_balance",
            "ppo_surrogate": "(compute+rho*communication)/(1+rho)",
        },
        "policy": {
            "state_dim": int(policy_cfg.state_dim),
            "horizon": int(policy_cfg.horizon),
            "max_step_dim": int(policy_cfg.max_step_dim),
            "max_num_levels": int(policy_cfg.max_num_levels),
        },
        "ppo": asdict(ppo),
        "rollout_size": int(train_cfg.rollout_size),
        "ppo_mode": layerwise_ppo_mode,
        "entropy_regularization": layerwise_entropy_regularization,
        "termination": algorithm_termination,
        "evidence_tiers": {
            "F1": {
                "split": SEARCH_EVIDENCE_SPLIT,
                "example_count": int(online_probe_example_count),
                "trials_per_episode": int(
                    train_cfg.online_num_trials_per_step
                ),
                "baseline_trial_count": int(
                    getattr(train_cfg, "baseline_groups", 5)
                    * getattr(train_cfg, "baseline_trials_per_group", 3)
                ),
                "roles": ["ppo_reward", "advantage", "promotion_prefilter"],
                "authoritative": False,
            },
            "F4": {
                "split": SEARCH_EVIDENCE_SPLIT,
                "example_count": int(authoritative_validation_example_count),
                "bank_a_trial_count": int(
                    authoritative_validation_banks.bank_a.trial_count
                ),
                "bank_b_trial_count": int(
                    authoritative_validation_banks.bank_b.trial_count
                ),
                "bank_c_trial_count": int(
                    authoritative_validation_banks.bank_c.trial_count
                ),
                "promotion_pooled_trial_count": int(
                    authoritative_validation_banks.promotion_trial_count
                ),
                "final_pooled_trial_count": int(
                    authoritative_validation_banks.final_trial_count
                ),
                "hard_gate": (
                    "joint_six_point_plus_compute_and_communication_"
                    "counterfactual_six_point_v1"
                ),
                "bootstrap_probability_role": "diagnostic_tiebreak_only",
                "roles": ["strict_frontier", "final_selection"],
                "authoritative": True,
            },
        },
        "persistence_protocol": "stable_parent_lock_incremental_fingerprint_v2",
    }
    algorithm_contract = bind_policy_network_contract(
        algorithm_contract,
        policy_shape={
            "state_dim": int(policy_cfg.state_dim),
            "horizon": int(policy_cfg.horizon),
            "max_step_dim": int(policy_cfg.max_step_dim),
            "max_num_levels": int(policy_cfg.max_num_levels),
            "d_model": int(policy_cfg.d_model),
            "n_heads": int(policy_cfg.n_heads),
            "n_layers": int(policy_cfg.n_layers),
            "d_ff": int(policy_cfg.d_ff),
            "dropout": float(policy_cfg.dropout),
            "actor_dim": int(policy_cfg.actor_dim),
            "critic_dim": int(policy_cfg.critic_dim),
            "mlp_critic_hidden": [512, 512, int(policy_cfg.d_model)],
        },
    )
    rl_variant = str(algorithm_contract["rl_variant"])
    algorithm_contract_hash = sha256_json(algorithm_contract)
    identity_context = _build_layerwise_candidate_identity_context(
        train_cfg=train_cfg,
        evaluator=evaluator,
        fusion_map=fusion_map,
        max_sfs=max_sfs,
        fixed_gelu=fixed_gelu,
        fixed_softmax=fixed_softmax,
        robust_reference=robust_reference,
        authoritative_robust_reference=authoritative_robust_reference,
        validation_banks=authoritative_validation_banks,
        probe_example_count=int(online_probe_example_count),
        authoritative_example_count=int(authoritative_validation_example_count),
        schedule=layerwise_env.schedule,
        static_skeletons_baseline=static_skeletons_baseline,
        algorithm_contract=algorithm_contract,
        algorithm_contract_hash=algorithm_contract_hash,
    )
    run_context = build_layerwise_run_context(
        identity_context,
        algorithm_contract_hash,
        {
            "dataset_protocol_schema": DATASET_PROTOCOL_SCHEMA,
            "dataset_protocol_hash": getattr(
                evaluator, "dataset_protocol_hash", None
            ),
            "search_split": SEARCH_EVIDENCE_SPLIT,
            "online_trials_per_episode": int(
                train_cfg.online_num_trials_per_step
            ),
            "promotion_validation_trials": int(
                getattr(train_cfg, "promotion_validation_trials", 15)
            ),
            "final_selection_validation_trials": int(
                getattr(train_cfg, "final_selection_validation_trials", 15)
            ),
            "baseline_groups": int(getattr(train_cfg, "baseline_groups", 5)),
            "baseline_trials_per_group": int(
                getattr(train_cfg, "baseline_trials_per_group", 3)
            ),
            "constraint_bootstrap_samples": int(
                getattr(train_cfg, "constraint_bootstrap_samples", 4096)
            ),
            "online_constraint_probability": float(
                getattr(train_cfg, "online_constraint_probability", 0.50)
            ),
            "promotion_constraint_probability": float(
                getattr(train_cfg, "promotion_constraint_probability", 0.80)
            ),
            "final_constraint_probability": float(
                getattr(train_cfg, "final_constraint_probability", 0.95)
            ),
            "validation_banks": authoritative_validation_banks.contract_payload(),
            "evidence_tiers": {
                "F1": {
                    "split": SEARCH_EVIDENCE_SPLIT,
                    "example_count": int(online_probe_example_count),
                    "fidelity": "F1",
                    "baseline_reference": dict(
                        baseline_preflight_metrics.get("robust_reference") or {}
                    ),
                },
                "F4": {
                    "split": SEARCH_EVIDENCE_SPLIT,
                    "example_count": int(authoritative_validation_example_count),
                    "fidelity": "F4",
                    "baseline_reference": dict(authoritative_robust_summary or {}),
                },
            },
        },
    )
    run_context_hash = sha256_json(run_context)
    run_lock.bind_context(run_context_hash)
    run_manifest = {
        "schema_version": LAYERWISE_RUN_SCHEMA,
        "dataset_protocol_schema": DATASET_PROTOCOL_SCHEMA,
        "dataset_protocol_hash": getattr(
            evaluator, "dataset_protocol_hash", None
        ),
        "search_split": SEARCH_EVIDENCE_SPLIT,
        "status": "running",
        "rl_variant": rl_variant,
        "policy_network_variant": policy_network_id,
        "algorithm_revision": algorithm_revision,
        "algorithm_contract": algorithm_contract,
        "algorithm_contract_hash": algorithm_contract_hash,
        "run_context": run_context,
        "run_context_hash": run_context_hash,
        "profile": str(train_cfg.profile),
        "decision_granularity": "layer",
        "reward_design": "robust_constrained",
        "fixed_gelu": [int(value) for value in np.asarray(fixed_gelu).reshape(-1)],
        "fixed_softmax": [int(value) for value in np.asarray(fixed_softmax).reshape(-1)],
        "fixed_label": str(fixed_label),
        "fixed_source": str(fixed_source),
        "stage1_config_path": str(
            getattr(evaluator, "stage1_best_config_input_path", "") or ""
        ),
        "stage1_config_sha256": str(
            getattr(evaluator, "stage1_best_config_input_sha256", "") or ""
        ),
        "planned_episodes": layerwise_termination["episode_limit"],
        "entropy_regularization": layerwise_entropy_regularization,
        "termination": layerwise_termination,
        "evidence_tiers": algorithm_contract["evidence_tiers"],
        "baseline_references": {
            "F1": dict(baseline_preflight_metrics.get("robust_reference") or {}),
            "F4": dict(authoritative_robust_summary or {}),
        },
        "ppo_mode": layerwise_ppo_mode,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    torch.manual_seed(int(train_cfg.seed))
    np.random.seed(int(train_cfg.seed) % (2**32))
    random.seed(int(train_cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = BLBStage2LayerwisePolicy(policy_cfg).to(device)
    policy_network_summary = policy.network_parameter_summary()
    run_manifest["policy_network"] = policy_network_summary
    log(
        f"  {bullet} Stage-2 policy network: {policy_network_id} "
        f"(total={policy_network_summary['total']:,}, "
        f"shared={policy_network_summary['shared']:,}, "
        f"actor_only={policy_network_summary['actor_only']:,}, "
        f"critic_only={policy_network_summary['critic_only']:,})"
    )
    initialize_layerwise_policy(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(train_cfg.ppo.lr))
    save_path = os.path.join(blb_progress_dir, "blb_stage2_rl_checkpoint_live.pt")
    candidate_store_path = os.path.join(blb_progress_dir, "candidate_store.jsonl")
    effective_resume_path = resume_checkpoint_path
    if not effective_resume_path and os.path.isfile(save_path):
        effective_resume_path = save_path
        log(f"  {bullet} 检测到 layerwise live checkpoint，自动 resume: {save_path}")
    start_episode = 0
    resumed_best: Dict[str, Any] = {}
    resumed_strict_pareto_frontier: List[Dict[str, Any]] = []
    resumed_candidate_store_size: Optional[int] = None
    resumed_diagnostics_jsonl_sizes: Optional[Mapping[str, Any]] = None
    resumed_store_file_fingerprints: Optional[Mapping[str, Any]] = None
    resumed_structured_run_id: Optional[str] = None
    resumed_ppo_update_count = 0
    resume_checkpoint: Optional[Mapping[str, Any]] = None
    cuda_rng_role_registry: List[Any] = []
    resumed_active_cuda_rng_states: Optional[List[Any]] = None
    planned_total_episodes = requested_total_episodes
    if effective_resume_path and os.path.isfile(effective_resume_path):
        try:
            checkpoint = torch.load(
                effective_resume_path, map_location=device, weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(effective_resume_path, map_location=device)
        validate_checkpoint_policy_network(checkpoint)
        validate_layerwise_checkpoint_metadata(
            checkpoint,
            rl_variant=rl_variant,
            algorithm_revision=algorithm_revision,
            algorithm_contract_hash=algorithm_contract_hash,
            run_context_hash=run_context_hash,
            dataset_protocol_schema=DATASET_PROTOCOL_SCHEMA,
            dataset_protocol_hash=getattr(
                evaluator, "dataset_protocol_hash", None
            ),
        )
        active_cuda_role_count = (
            int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        )
        cuda_rng_role_registry, resumed_active_cuda_rng_states = (
            resolve_cuda_rng_role_registry(
                checkpoint,
                active_role_count=active_cuda_role_count,
                new_role_state_factory=lambda role_index: (
                    torch.Generator(device=f"cuda:{role_index}")
                    .manual_seed(int(train_cfg.seed))
                    .get_state()
                    .cpu()
                ),
            )
        )
        resume_checkpoint = checkpoint
        start_episode = int(checkpoint.get("episode", 0))
        resumed_best = dict(checkpoint.get("strict_best") or {})
        if checkpoint.get("strict_pareto_frontier") is None:
            raise RuntimeError(
                "layerwise checkpoint strict resource Pareto frontier is missing"
            )
        resumed_strict_pareto_frontier = [
            dict(row) for row in checkpoint["strict_pareto_frontier"]
        ]
        resumed_candidate_store_size = checkpoint.get("candidate_store_size")
        resumed_diagnostics_jsonl_sizes = checkpoint.get("diagnostics_jsonl_sizes")
        resumed_store_file_fingerprints = checkpoint.get("store_file_fingerprints")
        resumed_structured_run_id = str(
            checkpoint.get("structured_run_id", "") or ""
        )
        if checkpoint.get("ppo_update_count") is None:
            raise RuntimeError("layerwise checkpoint PPO update count is missing")
        resumed_ppo_update_count = int(checkpoint.get("ppo_update_count"))
        if resumed_ppo_update_count < 0:
            raise RuntimeError("layerwise checkpoint PPO update count is invalid")
        checkpoint_planned_total = int(checkpoint.get(
            "planned_total_episodes", planned_total_episodes,
        ))
        validate_layerwise_episode_limit_extension(
            checkpoint_planned_total, planned_total_episodes,
        )
        log(f"  {bullet} layerwise resume @ episode {start_episode}")
    remaining_episode_budget = resolve_layerwise_episode_budget(
        requested_total_episodes,
        start_episode,
    )
    layerwise_train_cfg = LayerwiseTrainConfig(
        total_episodes=remaining_episode_budget,
        update_every_n_episodes=max(1, int(train_cfg.rollout_size)),
        log_every_n_episodes=max(1, int(train_cfg.rollout_size)),
        seed=int(train_cfg.seed),
        absolute_episode_start=int(start_episode),
        planned_total_episodes=int(planned_total_episodes),
        ppo=ppo,
        online_num_trials_per_step=int(train_cfg.online_num_trials_per_step),
        terminal_eval_batch_size=int(train_cfg.terminal_eval_batch_size),
        promotion_validation_trials=int(
            getattr(train_cfg, "promotion_validation_trials", 15)
        ),
        final_selection_validation_trials=int(
            getattr(train_cfg, "final_selection_validation_trials", 15)
        ),
        online_constraint_probability=float(
            getattr(train_cfg, "online_constraint_probability", 0.50)
        ),
        promotion_constraint_probability=float(
            getattr(train_cfg, "promotion_constraint_probability", 0.80)
        ),
        final_constraint_probability=float(
            getattr(train_cfg, "final_constraint_probability", 0.95)
        ),
        communication_importance_ratio=float(
            layerwise_env.communication_importance_ratio
        ),
    )
    candidate_store = CandidateStore(candidate_store_path)
    from rfr.common.jsonl_utils import iter_jsonl

    diagnostics_dir = os.path.join(blb_progress_dir, "diagnostics")
    existing_episode_path = os.path.join(diagnostics_dir, "episodes.jsonl")
    existing_update_path = os.path.join(diagnostics_dir, "ppo_updates.jsonl")

    repo_root = str(Path(__file__).resolve().parents[5])
    run_id_marker = os.path.join(blb_progress_dir, "rl_data_points_run_id.txt")
    if resume_checkpoint is None:
        validate_fresh_layerwise_run_state(
            run_id_marker,
            (
                candidate_store.path,
                existing_episode_path,
                existing_update_path,
                layerwise_manifest_path,
            ),
        )
    if os.path.isfile(run_id_marker):
        with open(run_id_marker, encoding="utf-8") as handle:
            structured_run_id = handle.read().strip()
    elif resume_checkpoint is not None:
        raise RuntimeError(
            "layerwise checkpoint structured run-id marker is missing; "
            "restore the complete run directory or start a fresh run"
        )
    else:
        run_source = (
            str(getattr(evaluator, "run_output_dir", "") or "").strip()
            or os.path.dirname(os.path.normpath(blb_progress_dir))
        )
        try:
            run_id_base = os.path.relpath(run_source, repo_root)
        except ValueError:
            run_id_base = run_source
        structured_run_id = make_unique_run_id(run_id_base)
        marker_tmp = run_id_marker + ".tmp"
        with open(marker_tmp, "w", encoding="utf-8") as handle:
            handle.write(structured_run_id + "\n")
        os.replace(marker_tmp, run_id_marker)
    layerwise_model_type = resolve_stage2_model_type(
        str(getattr(evaluator, "model_type", "") or ""),
        num_layers=layerwise_horizon,
    )
    stage2_data_writer = RLDataPointWriter(
        root_dir=os.path.join(blb_progress_dir, "records"),
        run_id=structured_run_id,
        stage="stage2",
        model_type=layerwise_model_type,
        dataset=str(train_cfg.profile),
    )

    def layerwise_slots_view(action_vec):
        from rfr.search.common.action_io import action_vec_to_slots_list

        return action_vec_to_slots_list(
            action_vec,
            max_sfs=max_sfs,
            num_layers=int(evaluator.total_layers),
            gelu_degree=fixed_gelu,
            attn_degree=fixed_softmax,
            profile=str(train_cfg.profile),
        )

    diag_recorder = RLDiagnosticsRecorder(
        output_dir=blb_progress_dir,
        num_layers=int(evaluator.total_layers),
        num_action_slots=int(getattr(base_env, "total_action_dim", 0) or 0),
        max_action_levels=64,
        top_k=20,
        log_fn=log,
        slots_view_builder=layerwise_slots_view,
        schema_version="stage2_layerwise_action_hml_v3",
        data_point_writer=stage2_data_writer,
        strict_writes=True,
        history_window=600,
        ppo_history_window=10,
    )

    def checkpoint_file_specs(
            candidate_size: Any,
            diagnostics_sizes: Any,
            ) -> Dict[str, Tuple[Any, int]]:
        if candidate_size is None:
            raise RuntimeError("layerwise checkpoint candidate_store_size is missing")
        sizes = dict(diagnostics_sizes or {})
        primary = dict(sizes.get("primary") or {})
        structured = dict(sizes.get("structured") or {})
        specs: Dict[str, Tuple[Any, int]] = {
            "candidate_store.jsonl": (candidate_store.path, int(candidate_size)),
            "primary/episodes.jsonl": (
                diag_recorder.episodes_path,
                int(primary.get("episodes.jsonl", 0)),
            ),
            "primary/ppo_updates.jsonl": (
                diag_recorder.ppo_updates_path,
                int(primary.get("ppo_updates.jsonl", 0)),
            ),
            "structured/episodes.jsonl": (
                stage2_data_writer.jsonl_path("episodes.jsonl"),
                int(structured.get("episodes.jsonl", 0)),
            ),
            "structured/ppo_updates.jsonl": (
                stage2_data_writer.jsonl_path("ppo_updates.jsonl"),
                int(structured.get("ppo_updates.jsonl", 0)),
            ),
        }
        return specs

    fingerprint_tracker = CheckpointFileFingerprintTracker()
    if resume_checkpoint is not None:
        if resumed_structured_run_id != structured_run_id:
            raise RuntimeError(
                "layerwise checkpoint structured run-id mismatch; "
                "restore the complete run directory or start a fresh run"
            )
        resume_file_specs = checkpoint_file_specs(
            resumed_candidate_store_size,
            resumed_diagnostics_jsonl_sizes,
        )
        fingerprint_tracker.validate_and_seed(
            dict(resumed_store_file_fingerprints or {}),
            resume_file_specs,
        )
        policy.load_state_dict(resume_checkpoint["policy"])
        if resume_checkpoint.get("policy_ppo_aux") is not None:
            policy.load_ppo_aux_state_dict(resume_checkpoint["policy_ppo_aux"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        candidate_store.recover_to_checkpoint_size(
            int(resumed_candidate_store_size),
        )
    diag_recorder.recover_to_checkpoint_sizes(
        resumed_diagnostics_jsonl_sizes,
    )
    if resume_checkpoint is not None:
        if resume_checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(resume_checkpoint["torch_rng_state"].cpu())
        if resumed_active_cuda_rng_states is not None:
            for role_index, state in enumerate(resumed_active_cuda_rng_states):
                torch.cuda.set_rng_state(state.cpu(), device=role_index)
        if resume_checkpoint.get("numpy_rng_state") is not None:
            np.random.set_state(resume_checkpoint["numpy_rng_state"])
        if resume_checkpoint.get("python_rng_state") is not None:
            random.setstate(resume_checkpoint["python_rng_state"])
    write_strict_json_file(layerwise_manifest_path, run_manifest)
    diag_recorder.set_baseline_action_vec(layerwise_env.pending_full_vector)
    restored_diagnostics = diag_recorder.restore_existing()
    expected_episode_high_water = int(start_episode) - 1
    if (
            int(restored_diagnostics["episodes"]) != int(start_episode)
            or int(diag_recorder.episode_high_water)
            != expected_episode_high_water
    ):
        raise RuntimeError(
            "layerwise checkpoint episode diagnostics mismatch: "
            f"checkpoint_count={start_episode}, "
            f"restored_count={restored_diagnostics['episodes']}, "
            f"restored_high_water={diag_recorder.episode_high_water}"
        )
    if (
            int(restored_diagnostics["ppo_updates"])
            != int(resumed_ppo_update_count)
            or int(diag_recorder.ppo_update_high_water)
            != int(resumed_ppo_update_count)
    ):
        raise RuntimeError(
            "layerwise checkpoint PPO diagnostics mismatch: "
            f"checkpoint_count={resumed_ppo_update_count}, "
            f"restored_count={restored_diagnostics['ppo_updates']}, "
            f"restored_high_water={diag_recorder.ppo_update_high_water}"
        )
    probability_thresholds = {
        "online": float(getattr(train_cfg, "online_constraint_probability", 0.50)),
        "promotion": float(getattr(train_cfg, "promotion_constraint_probability", 0.80)),
        "final": float(getattr(train_cfg, "final_constraint_probability", 0.95)),
    }
    constraint_limits = {
        "loss": float(robust_reference.loss_limit),
        "metric1": float(robust_reference.metric1_limit),
        "metric2": float(robust_reference.metric2_limit),
        "loss_std": float(robust_reference.loss_std_limit),
        "metric1_std": float(robust_reference.metric1_std_limit),
        "metric2_std": float(robust_reference.metric2_std_limit),
    }
    diag_recorder.set_meta({
        "profile": str(train_cfg.profile),
        "fixed_label": str(fixed_label),
        "fixed_source": str(fixed_source),
        "rl_variant": rl_variant,
        "policy_network_variant": policy_network_id,
        "policy_network": policy_network_summary,
        "decision_granularity": "layer",
        "reward_design": "robust_constrained",
        "algorithm_revision": algorithm_revision,
        "algorithm_contract_hash": algorithm_contract_hash,
        "run_context_hash": run_context_hash,
        "cost_model_revision": LAYERWISE_COST_MODEL_REVISION,
        "resource_objective": dict(algorithm_contract["resource_objective"]),
        "communication_importance_ratio": float(
            algorithm_contract["communication_importance_ratio"]
        ),
        "network_axis_weights": list(
            algorithm_contract["network_axis_weights"]
        ),
        "compute_axis_denominator": int(
            algorithm_contract["compute_axis_denominator"]
        ),
        "communication_axis_denominator": int(
            algorithm_contract["communication_axis_denominator"]
        ),
        "resource_credit_mode": algorithm_contract["resource_credit_mode"],
        "strict_resource_order": list(
            algorithm_contract["strict_resource_order"]
        ),
        "total_episodes_planned": layerwise_termination["episode_limit"],
        "rollout_size": int(train_cfg.rollout_size),
        "ppo_lr": float(train_cfg.ppo.lr),
        "gamma": 1.0,
        "gae_lambda": 1.0,
        "entropy_regularization": layerwise_entropy_regularization,
        "termination": layerwise_termination,
        "ppo_mode": layerwise_ppo_mode,
        "stage2_k_trials": int(train_cfg.online_num_trials_per_step),
        "baseline_groups": int(getattr(train_cfg, "baseline_groups", 5)),
        "baseline_trials_per_group": int(
            getattr(train_cfg, "baseline_trials_per_group", 3)
        ),
        "constraint_bootstrap_samples": int(
            getattr(train_cfg, "constraint_bootstrap_samples", 4096)
        ),
        "constraint_probabilities": probability_thresholds,
        "constraint_limits": constraint_limits,
        "baseline_preflight_metrics": dict(baseline_preflight_metrics),
        "borderline_retest_enabled": False,
        "borderline_retest_trials_multiplier": 1,
    })
    log(f"  {bullet} [data-points] layerwise Stage-2 → {stage2_data_writer.run_dir}")

    recent_episode_window = max(1, int(train_cfg.rollout_size))
    recent_episode_outcomes = deque(
        diag_recorder.recent_episode_outcomes(recent_episode_window),
        maxlen=recent_episode_window,
    )
    completed_episode_count = int(start_episode)
    best_reward_so_far = resolve_resumed_best_reward(
        resumed_best, diag_recorder.best_episode_return
    )
    best_selection_key: Optional[StrictSelectionKey] = None
    strict_best: Dict[str, Any] = dict(resumed_best)
    strict_pareto_frontier: List[Dict[str, Any]] = copy.deepcopy(
        resumed_strict_pareto_frontier
    )
    best_selection_key = strict_selection_key_from_snapshot(resumed_best)
    ppo_update_counter = int(resumed_ppo_update_count)
    started_at = time.time()

    def build_reloadable_best_group(best_payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        action_vec = best_payload.get("full_vector")
        if action_vec is None:
            return None
        fixed_config = build_fusion_fixed_config(
            action_vec,
            profile=str(train_cfg.profile),
            num_layers=int(evaluator.total_layers),
            gelu=np.asarray(fixed_gelu, dtype=int),
            softmax=np.asarray(fixed_softmax, dtype=int),
            fusion_map=fusion_map,
            source="stage2_layerwise_strict_best",
        )
        group = dict(fixed_config["group"])
        group["policy_actions"] = best_payload.get("action_matrix")
        overrides = best_payload.get("boosted_overrides") or {}
        group["boosted_overrides"] = [
            {
                "block_idx": int(block_idx),
                "layer_idx": int(layer_idx),
                "field_values": {
                    str(name): int(value) for name, value in values.items()
                },
            }
            for (block_idx, layer_idx), values in sorted(
                overrides.items(),
                key=lambda item: (int(item[0][1]), int(item[0][0])),
            )
        ]
        return group

    def write_strict_best_diagnostics(
            best_payload: Mapping[str, Any],
            *,
            episode: int,
            ) -> None:
        full_vector = best_payload.get("full_vector")
        if full_vector is None or len(full_vector) == 0:
            diag_recorder.clear_best_action_snapshot()
            return
        action_matrix = [list(row) for row in (best_payload.get("action_matrix") or [])]
        b4_count = sum(int(row[0]) for row in action_matrix if row)
        reward = float(best_payload.get("reward") or 0.0)
        variable_cost = float(best_payload.get("variable_cost") or 0.0)
        metrics = dict(best_payload.get("metrics") or {})
        diag_recorder.write_best_action_snapshot(
            episode_stats=EpisodeStats(
                episode=int(episode),
                total_reward=reward,
                terminal_reward=reward,
                per_step_sum=0.0,
                valid_steps=layerwise_horizon,
                invalid_steps=0,
                steps_taken=layerwise_horizon,
                total_bits=0,
                fusion_count=2 * layerwise_horizon + b4_count,
                first_invalid_step=None,
                first_invalid_block=None,
                first_invalid_layer=None,
                early_terminated=False,
                fusion_count_b2=layerwise_horizon,
                fusion_count_b4=b4_count,
                fusion_count_b5=layerwise_horizon,
                terminal_priority=3,
                terminal_loss_mean=float(metrics.get("loss_mean", 0.0)),
                terminal_loss_std=float(metrics.get("loss_std", 0.0)),
                terminal_metric1_mean=float(metrics.get("metric1_mean", 0.0)),
                terminal_metric1_std=float(metrics.get("metric1_std", 0.0)),
                terminal_metric2_mean=float(metrics.get("metric2_mean", 0.0)),
                terminal_metric2_std=float(metrics.get("metric2_std", 0.0)),
                terminal_cost_score=variable_cost,
                terminal_cost_rank_score=variable_cost,
                variable_cost=variable_cost,
                compute_saving=float(best_payload.get("compute_saving") or 0.0),
                communication_saving=float(
                    best_payload.get("communication_saving") or 0.0
                ),
                robust_floor=float(best_payload.get("robust_floor") or 0.0),
                secondary_progress=float(
                    best_payload.get("secondary_progress") or 0.0
                ),
                ppo_resource_score=float(
                    best_payload.get("ppo_resource_score") or variable_cost
                ),
                compute_shapley_credit=float(
                    best_payload.get("compute_shapley_credit") or 0.0
                ),
                communication_shapley_credit=float(
                    best_payload.get("communication_shapley_credit") or 0.0
                ),
                layer_resource_rewards=list(
                    best_payload.get("layer_resource_rewards") or []
                ),
                slot_resource_rewards=list(
                    best_payload.get("slot_resource_rewards") or []
                ),
                layer_action_matrix=action_matrix,
                promotion_status="strict_best_reconciled",
            ),
            full_action_vec=np.asarray(full_vector, dtype=np.int64),
            best_reward_so_far=reward,
        )

    def save_layerwise_checkpoint(
            *,
            completed: int,
            strict_best: Optional[Mapping[str, Any]],
            ) -> None:
        nonlocal cuda_rng_role_registry
        best_payload = dict(strict_best or {})
        checkpoint_best_action = best_payload.get("full_vector")
        checkpoint_best_group = build_reloadable_best_group(best_payload)
        candidate_store_size = (
            candidate_store.path.stat().st_size
            if candidate_store.path.exists() else 0
        )
        diagnostics_jsonl_sizes = diag_recorder.committed_jsonl_sizes()
        store_file_fingerprints = fingerprint_tracker.fingerprints(
            checkpoint_file_specs(candidate_store_size, diagnostics_jsonl_sizes)
        )
        active_cuda_rng_states = (
            [
                state.cpu()
                for state in torch.cuda.get_rng_state_all()
            ]
            if torch.cuda.is_available()
            else []
        )
        cuda_rng_role_registry = merge_cuda_rng_role_registry(
            cuda_rng_role_registry,
            active_cuda_rng_states,
        )
        checkpoint = {
            "dataset_protocol_schema": DATASET_PROTOCOL_SCHEMA,
            "dataset_protocol_hash": getattr(
                evaluator, "dataset_protocol_hash", None
            ),
            "policy": policy.state_dict(),
            "policy_ppo_aux": policy.ppo_aux_state_dict(),
            "optimizer": optimizer.state_dict(),
            "episode": int(completed),
            "strict_best": best_payload,
            "strict_pareto_frontier": copy.deepcopy(strict_pareto_frontier),
            "best_action": checkpoint_best_action,
            "blb_v3_best_action_vec": checkpoint_best_action,
            "blb_v3_best_action_group": checkpoint_best_group,
            "blb_v3_fusion_count_action": True,
            "profile": str(train_cfg.profile),
            "planned_total_episodes": int(planned_total_episodes),
            "candidate_store_size": int(candidate_store_size),
            "diagnostics_jsonl_sizes": diagnostics_jsonl_sizes,
            "store_file_fingerprints": store_file_fingerprints,
            "structured_run_id": structured_run_id,
            "ppo_update_count": int(ppo_update_counter),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": active_cuda_rng_states or None,
            "cuda_rng_role_registry_version": 1,
            "cuda_rng_state_by_role": cuda_rng_role_registry,
            "cuda_rng_active_role_count": len(active_cuda_rng_states),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "rl_variant": rl_variant,
            "policy_network_variant": policy_network_id,
            "policy_network": policy_network_summary,
            "algorithm_revision": algorithm_revision,
            "algorithm_contract": algorithm_contract,
            "algorithm_contract_hash": algorithm_contract_hash,
            "run_context": run_context,
            "run_context_hash": run_context_hash,
        }
        tmp_path = save_path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, save_path)

    if resume_checkpoint is None:
        save_layerwise_checkpoint(
            completed=0,
            strict_best=strict_best,
        )

    def on_layerwise_episode(record: Any) -> None:
        nonlocal best_selection_key, completed_episode_count, best_reward_so_far
        episode_identity = int(record.episode_index)
        if episode_identity != completed_episode_count:
            raise RuntimeError(
                "layerwise episode callback identity mismatch: "
                f"expected={completed_episode_count}, received={episode_identity}"
            )
        if episode_identity != diag_recorder.episode_high_water + 1:
            raise RuntimeError(
                "layerwise episode diagnostics identity mismatch: "
                f"high_water={diag_recorder.episode_high_water}, "
                f"received={episode_identity}"
            )
        completed_episode_count += 1
        best_reward_so_far = max(best_reward_so_far, float(record.reward))
        recent_episode_outcomes.append((
            float(record.reward),
            int(record.invalid_steps),
        ))
        pooled_assessment = _to_plain_mapping(record.assessment)
        fresh_assessment = _to_plain_mapping(record.fresh_assessment)
        fresh_metrics = dict(record.metrics or {})
        pooled_metrics = dict(record.pooled_metrics or {})
        probe_diagnostics = _to_plain_mapping(record.probe_diagnostics)

        def trials_payload(trials: Any) -> Dict[str, Any]:
            if trials is None:
                return {}
            return {
                "loss": [float(value) for value in trials.loss],
                "metric1": [float(value) for value in trials.metric1],
                "metric2": [float(value) for value in trials.metric2],
                "seeds": [int(value) for value in trials.seeds],
            }

        fresh_trials = trials_payload(record.raw_trials)
        pooled_trials = trials_payload(record.pooled_trials)
        fresh_probabilities = {
            name: float(fresh_assessment[name])
            for name in _PROBABILITY_FIELDS if name in fresh_assessment
        }
        pooled_probabilities = {
            name: float(pooled_assessment[name])
            for name in _PROBABILITY_FIELDS if name in pooled_assessment
        }
        promotion_assessment = _to_plain_mapping(record.promotion_assessment)
        promotion_metrics = _to_plain_mapping(record.promotion_metrics)
        promotion_probabilities = {
            name: float(promotion_assessment[name])
            for name in _PROBABILITY_FIELDS if name in promotion_assessment
        }
        b4_count = sum(int(row[0]) for row in record.action_matrix)
        k_values = [
            int(k_value)
            for row in record.action_matrix
            for k_value in PRECISION_PRESETS[int(row[1])].simulation_k_by_block
        ]
        avg_k = float(np.mean(k_values)) if k_values else 13.0
        is_new_best = False
        if (
                record.promotion_status in ("promoted", "already_promoted")
                and record.promotion_candidate_key
                and len(promotion_probabilities) == len(_PROBABILITY_FIELDS)
        ):
            selection_key = strict_selection_key(
                record.promotion_candidate_key,
                {
                    "variable_cost": record.variable_cost,
                    "compute_saving": record.compute_saving,
                    "communication_saving": record.communication_saving,
                    "communication_importance_ratio": float(
                        layerwise_env.communication_importance_ratio
                    ),
                    "robust_floor": record.robust_floor,
                    "secondary_progress": record.secondary_progress,
                    "action_matrix": record.action_matrix,
                    "assessment": promotion_assessment,
                    "metrics": promotion_metrics,
                    "constraint_safety_margins": (
                        normalized_constraint_safety_margins(
                            promotion_metrics,
                            authoritative_robust_reference,
                        )
                    ),
                    "full_vector": record.pending_full_vector,
                },
            )
            if best_selection_key is None or selection_key < best_selection_key:
                best_selection_key = selection_key
                is_new_best = True
        episode_stats = EpisodeStats(
                episode=int(record.episode_index),
                total_reward=float(record.reward),
                terminal_reward=float(record.reward),
                per_step_sum=0.0,
                valid_steps=layerwise_horizon - int(record.invalid_steps),
                invalid_steps=int(record.invalid_steps),
                steps_taken=layerwise_horizon,
                total_bits=0,
                fusion_count=2 * layerwise_horizon + b4_count,
                first_invalid_step=None,
                first_invalid_block=None,
                first_invalid_layer=None,
                early_terminated=False,
                fusion_count_b2=layerwise_horizon,
                fusion_count_b4=b4_count,
                fusion_count_b5=layerwise_horizon,
                terminal_final_config_fingerprint=str(
                    record.final_config_fingerprint
                ),
                terminal_materialization_failure_reason=str(
                    record.materialization_failure_reason
                ),
                terminal_model_uses_replan_config=bool(
                    record.model_uses_replan_config
                ),
                terminal_priority=int(record.priority),
                terminal_loss_mean=float(fresh_metrics.get("loss_mean", 0.0)),
                terminal_loss_std=float(fresh_metrics.get("loss_std", 0.0)),
                terminal_metric1_mean=float(fresh_metrics.get("metric1_mean", 0.0)),
                terminal_metric1_std=float(fresh_metrics.get("metric1_std", 0.0)),
                terminal_metric2_mean=float(fresh_metrics.get("metric2_mean", 0.0)),
                terminal_metric2_std=float(fresh_metrics.get("metric2_std", 0.0)),
                terminal_k_gain=13.0 - avg_k,
                terminal_fusion_gain=(
                    float(b4_count) / float(layerwise_horizon)
                ),
                terminal_cost_score=float(record.variable_cost),
                terminal_cost_rank_score=float(record.variable_cost),
                terminal_probe_wall_seconds=float(
                    probe_diagnostics.get("wall_seconds", 0.0) or 0.0
                ),
                terminal_probe_devices=[str(value) for value in (probe_diagnostics.get("devices") or [])],
                terminal_probe_trial_counts=[
                    int(value) for value in (
                        probe_diagnostics.get("per_worker_trial_counts") or []
                    )
                ],
                terminal_probe_trial_indices=[
                    [int(index) for index in (indices or [])]
                    for indices in (
                        probe_diagnostics.get("per_worker_trial_indices") or []
                    )
                ],
                terminal_probe_speedup=float(
                    probe_diagnostics.get("speedup_vs_sequential", 1.0) or 1.0
                ),
                terminal_cost_eval_wall_seconds=float(
                    probe_diagnostics.get("cost_eval_wall_seconds", 0.0) or 0.0
                ),
                terminal_probe_install_wall_seconds=float(
                    probe_diagnostics.get("probe_install_wall_seconds", 0.0) or 0.0
                ),
                terminal_probe_clear_wall_seconds=float(
                    probe_diagnostics.get("probe_clear_wall_seconds", 0.0) or 0.0
                ),
                terminal_probe_install_skipped=bool(probe_diagnostics.get(
                    "probe_install_skipped", False
                )),
                terminal_probe_clear_skipped=bool(probe_diagnostics.get(
                    "probe_clear_skipped", False
                )),
                raw_trials=fresh_trials,
                constraint_probabilities=pooled_probabilities,
                fresh_trials=fresh_trials,
                pooled_trials=pooled_trials,
                fresh_metrics={str(k): float(v) for k, v in fresh_metrics.items()},
                pooled_metrics={str(k): float(v) for k, v in pooled_metrics.items()},
                fresh_constraint_probabilities=fresh_probabilities,
                pooled_constraint_probabilities=pooled_probabilities,
                fresh_trial_count=int(record.fresh_trial_count),
                pooled_trial_count=int(record.pooled_trial_count),
                reward_evidence=str(record.reward_evidence),
                ranking_evidence=str(record.ranking_evidence),
                constraint_thresholds={
                    **constraint_limits,
                    **probability_thresholds,
                },
                variable_cost=float(record.variable_cost),
                compute_saving=float(record.compute_saving),
                communication_saving=float(record.communication_saving),
                robust_floor=float(record.robust_floor),
                secondary_progress=float(record.secondary_progress),
                ppo_resource_score=float(record.ppo_resource_score),
                compute_shapley_credit=float(record.compute_shapley_credit),
                communication_shapley_credit=float(
                    record.communication_shapley_credit
                ),
                layer_resource_rewards=[
                    float(value) for value in record.layer_resource_rewards
                ],
                slot_resource_rewards=[
                    [float(value) for value in row]
                    for row in record.slot_resource_rewards
                ],
                layer_action_matrix=[list(row) for row in record.action_matrix],
                block4_entropy=record.block4_entropy,
                k_entropy=record.k_entropy,
                promotion_trial_count=int(record.promoted_trial_count),
                promotion_status=str(record.promotion_status),
            )
        diag_recorder.record_episode(
            episode_stats=episode_stats,
            full_action_vec=np.asarray(record.pending_full_vector, dtype=np.int64),
            is_new_best=is_new_best,
            best_reward_so_far=float(best_reward_so_far),
        )
        status.update_after_episode(
            int(record.episode_index) + 1,
            float(record.reward),
            {
                "priority": int(record.priority),
                "variable_cost": float(record.variable_cost),
                "compute_saving": float(record.compute_saving),
                "communication_saving": float(record.communication_saving),
                "robust_floor": float(record.robust_floor),
                "secondary_progress": float(record.secondary_progress),
                "ppo_resource_score": float(record.ppo_resource_score),
                "block4_entropy": record.block4_entropy,
                "k_entropy": record.k_entropy,
                "strict_revalidation_status": record.strict_revalidation_status,
                "termination_reason": record.termination_reason,
            },
        )

    def on_layerwise_update(metrics: Mapping[str, Any], completed: int, record: Any) -> None:
        nonlocal ppo_update_counter, strict_best, strict_pareto_frontier
        nonlocal best_selection_key
        if int(completed) != completed_episode_count:
            raise RuntimeError(
                "layerwise PPO callback episode count mismatch: "
                f"expected={completed_episode_count}, received={completed}"
            )
        ppo_update_counter += 1
        if ppo_update_counter != diag_recorder.ppo_update_high_water + 1:
            raise RuntimeError(
                "layerwise PPO diagnostics identity mismatch: "
                f"high_water={diag_recorder.ppo_update_high_water}, "
                f"received={ppo_update_counter}"
            )
        strict_best = dict(metrics.get("strict_best") or {})
        strict_pareto_frontier = [
            dict(row) for row in metrics.get("strict_pareto_frontier", [])
        ]
        best_selection_key = strict_selection_key_from_snapshot(strict_best)
        write_strict_best_diagnostics(strict_best, episode=int(record.episode_index))
        recent = list(recent_episode_outcomes)
        recent_rewards = [float(item[0]) for item in recent] or [0.0]
        update_stats = PPOUpdateStats(
            update=ppo_update_counter,
            completed_episodes=int(completed),
            policy_loss=float(metrics.get("policy_loss", 0.0)),
            value_loss=float(metrics.get("value_loss", 0.0)),
            entropy=float(metrics.get("entropy", 0.0)),
            clip_fraction=float(metrics.get("clip_fraction", 0.0)),
            n_samples=int(
                metrics.get("n_samples", len(recent) * layerwise_horizon)
            ),
            window_mean_return=float(np.mean(recent_rewards)),
            window_max_return=float(np.max(recent_rewards)),
            window_min_return=float(np.min(recent_rewards)),
            window_mean_invalid=float(np.mean([item[1] for item in recent])),
            best_reward_so_far=float(best_reward_so_far),
            elapsed_sec=float(time.time() - started_at),
            ent_coef=float(metrics.get("ent_coef", 0.0)),
            approx_kl=float(metrics.get("approx_kl", 0.0)),
            kl_early_stop=bool(metrics.get("kl_early_stop", False)),
            lr=float(metrics.get("lr", train_cfg.ppo.lr)),
            lr_scale=float(metrics.get("lr_scale", 1.0)),
            entropy_recovery_delta=float(
                metrics.get("entropy_recovery_delta", 0.0)
            ),
            nonfinite_minibatches=int(metrics.get("nonfinite_minibatches", 0) or 0),
            nonfinite_update_skipped=bool(
                metrics.get("nonfinite_update_skipped", False)
            ),
            return_mean=float(metrics.get("return_mean", 0.0)),
            return_std=float(metrics.get("return_std", 1.0)),
            block4_entropy=metrics.get("block4_entropy"),
            k_entropy=metrics.get("k_entropy"),
            strict_revalidation_status=str(
                metrics.get("strict_revalidation_status", "not_due")
            ),
            termination_reason=str(metrics.get("termination_reason", "running")),
            strict_pareto_frontier=[
                dict(row) for row in metrics.get("strict_pareto_frontier", [])
            ],
            actor_clip_mode=str(metrics.get("actor_clip_mode", "joint")),
            actor_credit_mode=str(metrics.get("actor_credit_mode", "scalar_gae")),
            entropy_objective_mode=str(
                metrics.get("entropy_objective_mode", "joint_sum")
            ),
            slot_labels=[str(value) for value in metrics.get("slot_labels", [])],
            entropy_per_slot=list(metrics.get("entropy_per_slot", [])),
            approx_kl_per_slot=list(metrics.get("approx_kl_per_slot", [])),
            clip_fraction_per_slot=list(metrics.get("clip_fraction_per_slot", [])),
            raw_advantage_mean_per_slot=list(
                metrics.get("raw_advantage_mean_per_slot", [])
            ),
            raw_advantage_std_per_slot=list(
                metrics.get("raw_advantage_std_per_slot", [])
            ),
            raw_advantage_snr_per_slot=list(
                metrics.get("raw_advantage_snr_per_slot", [])
            ),
            value_explained_variance_pre=metrics.get(
                "value_explained_variance_pre"
            ),
            value_explained_variance_post=metrics.get(
                "value_explained_variance_post"
            ),
            value_rmse_pre=metrics.get("value_rmse_pre"),
            value_rmse_post=metrics.get("value_rmse_post"),
            value_bias_pre=metrics.get("value_bias_pre"),
            value_bias_post=metrics.get("value_bias_post"),
            shared_grad_parameter_count=int(
                metrics.get("shared_grad_parameter_count", 0) or 0
            ),
            actor_shared_grad_norm=metrics.get("actor_shared_grad_norm"),
            critic_shared_grad_norm=metrics.get("critic_shared_grad_norm"),
            actor_critic_shared_grad_cosine=metrics.get(
                "actor_critic_shared_grad_cosine"
            ),
            preclip_grad_norm_mean=metrics.get("preclip_grad_norm_mean"),
            preclip_grad_norm_max=metrics.get("preclip_grad_norm_max"),
        )
        diag_recorder.record_ppo_update(update_stats)
        save_layerwise_checkpoint(
            completed=int(completed),
            strict_best=strict_best,
        )
        shared_probe_runner = getattr(
            base_env,
            "_shared_probe_runner_owner",
            None,
        )
        if shared_probe_runner is not None:
            deferred_gpu_failure = (
                shared_probe_runner.pop_deferred_gpu_failure()
            )
            if deferred_gpu_failure is not None:
                raise deferred_gpu_failure
        raise_if_elastic_gpu_restart_requested(
            work_remaining=int(completed) < int(planned_total_episodes),
        )
        status.update_after_ppo_update(
            int(ppo_update_counter),
            {
                "completed_episodes": int(completed),
                "policy_loss": float(update_stats.policy_loss),
                "value_loss": float(update_stats.value_loss),
                "entropy": float(update_stats.entropy),
                "clip_fraction": float(update_stats.clip_fraction),
                "ent_coef": update_stats.ent_coef,
                "approx_kl": float(update_stats.approx_kl),
                "kl_early_stop": update_stats.kl_early_stop,
                "lr": update_stats.lr,
                "lr_scale": update_stats.lr_scale,
                "entropy_recovery_delta": update_stats.entropy_recovery_delta,
                "nonfinite_minibatches": update_stats.nonfinite_minibatches,
                "nonfinite_update_skipped": update_stats.nonfinite_update_skipped,
                "return_mean": update_stats.return_mean,
                "return_std": update_stats.return_std,
                "value_explained_variance_post": (
                    update_stats.value_explained_variance_post
                ),
                "value_rmse_post": update_stats.value_rmse_post,
                "actor_critic_shared_grad_cosine": (
                    update_stats.actor_critic_shared_grad_cosine
                ),
                "preclip_grad_norm_mean": update_stats.preclip_grad_norm_mean,
                "entropy_per_slot": update_stats.entropy_per_slot,
                "approx_kl_per_slot": update_stats.approx_kl_per_slot,
                "clip_fraction_per_slot": update_stats.clip_fraction_per_slot,
                "window_mean_return": float(update_stats.window_mean_return),
                "window_max_return": float(update_stats.window_max_return),
                "window_min_return": float(update_stats.window_min_return),
                "window_mean_invalid": float(update_stats.window_mean_invalid),
                "block4_entropy": update_stats.block4_entropy,
                "k_entropy": update_stats.k_entropy,
                "strict_revalidation_status": (
                    update_stats.strict_revalidation_status
                ),
                "termination_reason": update_stats.termination_reason,
                "strict_pareto_frontier": update_stats.strict_pareto_frontier,
                "actor_clip_mode": update_stats.actor_clip_mode,
                "actor_credit_mode": update_stats.actor_credit_mode,
                "entropy_objective_mode": update_stats.entropy_objective_mode,
            },
        )
        if strict_best.get("reward") is not None and strict_best.get("full_vector"):
            best_full_vector = [int(value) for value in strict_best["full_vector"]]
            current_full_vector = [
                int(value) for value in record.pending_full_vector
            ]
            status.set_best(
                best_reward=float(strict_best["reward"]),
                best_action_vec=best_full_vector,
                best_breakdown={
                    "priority": 3,
                    "variable_cost": strict_best.get("variable_cost"),
                    "resource_objective": {
                        field_name: strict_best.get(field_name)
                        for field_name in (
                            "compute_saving",
                            "communication_saving",
                            "robust_floor",
                            "secondary_progress",
                            "ppo_resource_score",
                            "compute_shapley_credit",
                            "communication_shapley_credit",
                        )
                    },
                    "action_matrix": strict_best.get("action_matrix"),
                    "assessment": strict_best.get("assessment"),
                    "metrics": strict_best.get("metrics"),
                },
                best_episode=(
                    int(record.episode_index) + 1
                    if current_full_vector == best_full_vector else None
                ),
            )
        if int(completed) % max(1, int(train_cfg.save_interval)) == 0:
            diag_recorder.flush_periodic()
    status.set_phase(
        f"PPO training ({layerwise_horizon}-step layerwise robust)"
    )
    from rfr.search.runtime.control import (
        STOP_FLAG_FILENAME as NOISE_STAGE_STOP_FLAG_FILENAME,
        consume_stop_flag as consume_stop_flag_file,
        graceful_stop_requested as is_graceful_stop_requested,
        install_graceful_stop_handler,
        reset_graceful_stop_state,
        uninstall_graceful_stop_handler,
    )

    stop_flag_path = os.path.join(
        blb_progress_dir,
        NOISE_STAGE_STOP_FLAG_FILENAME,
    )
    reset_graceful_stop_state()
    consume_stop_flag_file(stop_flag_path)
    install_graceful_stop_handler(log_fn=log)
    log(
        f"  {bullet} [graceful-stop] Ctrl+C or create {stop_flag_path}; "
        "the layerwise run will stop after the next PPO checkpoint boundary."
    )
    training_completed = False
    completion_status = "failed"
    summary: Dict[str, Any]
    try:
        summary = train_layerwise(
            env=layerwise_env,
            promotion_base_env=promotion_base_env,
            validation_banks=authoritative_validation_banks,
            policy=policy,
            train_cfg=layerwise_train_cfg,
            candidate_store=candidate_store,
            identity_context=identity_context,
            device=device,
            optimizer=optimizer,
            on_episode_end=on_layerwise_episode,
            on_ppo_update_end=on_layerwise_update,
            stop_requested=lambda: is_graceful_stop_requested(stop_flag_path),
            retain_history=False,
        )
        summary_completed_episodes = int(summary.get(
            "completed_episodes", completed_episode_count,
        ))
        if summary_completed_episodes != completed_episode_count:
            raise RuntimeError(
                "layerwise training summary episode count mismatch: "
                f"callbacks={completed_episode_count}, "
                f"summary={summary_completed_episodes}"
            )
        if (
                diag_recorder.episode_count != completed_episode_count
                or diag_recorder.ppo_update_count != ppo_update_counter
        ):
            raise RuntimeError(
                "layerwise diagnostics cumulative count mismatch: "
                f"episodes={diag_recorder.episode_count}/{completed_episode_count}, "
                f"updates={diag_recorder.ppo_update_count}/{ppo_update_counter}"
            )
        strict_best = dict(summary.get("strict_best") or {})
        strict_pareto_frontier = [
            dict(row) for row in summary.get("strict_pareto_frontier", [])
        ]
        write_strict_best_diagnostics(
            strict_best,
            episode=int(completed_episode_count),
        )
        save_layerwise_checkpoint(
            completed=int(completed_episode_count),
            strict_best=summary.get("strict_best"),
        )
        if summary.get("graceful_stopped", False):
            completion_status = "graceful_stop"
        else:
            completion_status = "maximum_episodes"
        training_completed = True
    except ElasticGPUFailure:
        raise
    except Exception as exc:
        if not is_recoverable_gpu_failure(exc):
            raise
        raise ElasticGPUFailure(
            device="cuda:0",
            role="learner-primary",
            operation="stage2_layerwise_training",
            cause=exc,
        ) from exc
    finally:
        try:
            if not training_completed:
                status.set_phase("failed")
            run_manifest.update({
                "status": completion_status,
                "completed_episodes": int(completed_episode_count),
                "ppo_update_count": int(ppo_update_counter),
                "best_resource_objective": (
                    None
                    if not strict_best
                    else {
                        field_name: copy.deepcopy(strict_best.get(field_name))
                        for field_name in (
                            "compute_saving",
                            "communication_saving",
                            "robust_floor",
                            "secondary_progress",
                            "ppo_resource_score",
                        )
                    }
                ),
                "strict_pareto_frontier": copy.deepcopy(
                    strict_pareto_frontier
                ),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            write_strict_json_file(layerwise_manifest_path, run_manifest)
        finally:
            try:
                diag_recorder.finalize(status=completion_status)
            finally:
                try:
                    base_env.clear_installed_blb()
                finally:
                    try:
                        for restore_name in (
                                "restore_layer_block5_noise", "restore_layer_block4_noise",
                                "restore_layer_block3_noise", "restore_layer_block2_noise",
                                "restore_layer_block1_noise",
                        ):
                            method = getattr(
                                evaluator.reversible_handler, restore_name, None,
                            )
                            if method is None:
                                continue
                            try:
                                method(layer_indices=list(range(evaluator.total_layers)))
                            except Exception:
                                pass
                        evaluator.apply_configuration(fixed_gelu, fixed_softmax)
                    finally:
                        shared_owner = getattr(
                            base_env, "_shared_probe_runner_owner", None,
                        )
                        if shared_owner is not None:
                            shared_owner.close()
                        else:
                            promotion_runner = getattr(
                                promotion_base_env, "probe_runner", None,
                            )
                            if (
                                    promotion_runner is not None
                                    and promotion_runner is not getattr(
                                        base_env, "probe_runner", None,
                                    )
                            ):
                                promotion_runner.close()
    uninstall_graceful_stop_handler()
    status.set_phase(completion_status)

    bank_b_best = dict(summary.get("bank_b_best") or {})
    compact_summary = {
        "schema_version": "stage2_layerwise_robust_summary_v6",
        "status": completion_status,
        "rl_variant": rl_variant,
        "policy_network_variant": policy_network_id,
        "policy_network": policy_network_summary,
        "algorithm_revision": algorithm_revision,
        "algorithm_contract_hash": algorithm_contract_hash,
        "run_context_hash": run_context_hash,
        "communication_importance_ratio": float(
            layerwise_env.communication_importance_ratio
        ),
        "network_axis_weights": list(
            algorithm_contract["network_axis_weights"]
        ),
        "axis_precision_tolerances": list(
            algorithm_contract["axis_precision_tolerances"]
        ),
        "completed_episodes": int(summary.get("completed_episodes", start_episode)),
        "best_action_matrix": summary.get("best_action_matrix"),
        "best_layer_configurations": summary.get(
            "best_layer_configurations"
        ),
        "best_full_vector": summary.get("best_full_vector"),
        "best_assessment": summary.get("best_assessment"),
        "strict_best_assessment": summary.get("best_assessment"),
        "best_metrics": summary.get("best_metrics"),
        "best_resource_objective": summary.get("best_resource_objective"),
        "strict_pareto_frontier": summary.get("strict_pareto_frontier", []),

        "best_variable_cost": summary.get("best_variable_cost"),
        "best_reward": summary.get("best_reward"),
        "best_promotion_evidence": summary.get("best_promotion_evidence"),
        "best_axis_counterfactuals": summary.get(
            "best_axis_counterfactuals"
        ),
        "bank_b_best": bank_b_best or None,
        "final_evidence": {
            "status": (
                "strict_revalidation_passed"
                if summary.get("strict_revalidation_status") == "passed"
                else "bank_b_confirmed_not_final_certified"
                if bank_b_best
                else "no_candidate"
            ),
            "diagnostic_probability": float(
                getattr(train_cfg, "final_constraint_probability", 0.95)
            ),
            "hard_gate": (
                "joint_six_point_plus_compute_and_communication_"
                "counterfactual_six_point_v1"
            ),
            "bank_a_trial_count": int(
                authoritative_validation_banks.bank_a.trial_count
            ),
            "bank_b_trial_count": int(
                authoritative_validation_banks.bank_b.trial_count
            ),
            "bank_c_trial_count": int(
                authoritative_validation_banks.bank_c.trial_count
            ),
            "pooled_final_trial_count": int(
                authoritative_validation_banks.final_trial_count
            ),
            "current_assessment": (
                summary.get("best_assessment")
                or bank_b_best.get("assessment")
            ),
            "note": (
                "Bank A qualifies a candidate, independent Bank B confirms "
                "the pooled AB point gate, and held-out Bank C certifies the "
                "pooled ABC point gate. The same fixed banks certify the "
                "compute-only and communication-only counterfactuals against "
                "their allocated precision budgets; probabilities are "
                "diagnostics only."
            ),
        },
        "block4_entropy": summary.get("block4_entropy"),
        "k_entropy": summary.get("k_entropy"),
        "precision_preset_entropy": summary.get("k_entropy"),
        "strict_revalidation_status": str(
            summary.get("strict_revalidation_status", "not_due")
        ),
        "entropy_regularization": layerwise_entropy_regularization,
        "termination": layerwise_termination,
        "termination_reason": str(
            summary.get("termination_reason") or completion_status
        ),
        "evidence_tiers": algorithm_contract["evidence_tiers"],
        "constraint_probability_thresholds": probability_thresholds,
        "constraint_limits": constraint_limits,
        "baseline_reference": dict(baseline_preflight_metrics),
        "ppo_update_count": int(ppo_update_counter),
        "candidate_store": candidate_store.path,
        "checkpoint": save_path,
        "structured_data_dir": stage2_data_writer.run_dir,
    }
    compact_summary = to_jsonable(compact_summary, stringify_unknown=True)
    write_strict_json_file(
        os.path.join(blb_progress_dir, "layerwise_summary.json"),
        compact_summary,
    )
    stage2_data_writer.write_summary(compact_summary)
    stage2_data_writer.close()
    if summary.get("graceful_stopped", False):
        consume_stop_flag_file(stop_flag_path)
        status.mark_stopped(
            reason="checkpoint-boundary graceful stop",
            completed_episodes=int(completed_episode_count),
        )
        log(
            f"  [graceful-stop] checkpoint saved at episode "
            f"{int(completed_episode_count)} -> {save_path}; "
            "launch again with the same parameters to resume."
        )
        raise SystemExit(0)

    curve_series = {
        "returns": [], "loss": [], "metric1": [], "metric2": [],
        "fusion": [], "avg_k": [], "entropy": [], "entropy_episode": [],
    }
    if os.path.isfile(existing_episode_path):
        for row in iter_jsonl(existing_episode_path, errors="raise"):
            curve_series["returns"].append(float(row.get("total_reward", 0.0)))
            curve_series["loss"].append(float(row.get("terminal_loss_mean", 0.0)))
            curve_series["metric1"].append(float(row.get("terminal_metric1_mean", 0.0)))
            curve_series["metric2"].append(float(row.get("terminal_metric2_mean", 0.0)))
            curve_series["fusion"].append(int(row.get("fusion_count", 0)))
            curve_series["avg_k"].append(
                13.0 - float(row.get("terminal_k_gain", 0.0))
            )
    if os.path.isfile(existing_update_path):
        for row in iter_jsonl(existing_update_path, errors="raise"):
            curve_series["entropy"].append(float(row.get("entropy", 0.0)))
            curve_series["entropy_episode"].append(
                int(row.get("completed_episodes", 0))
            )
    if curve_series["returns"]:
        write_training_curves(
            blb_progress_dir,
            episode_returns=curve_series["returns"],
            episode_losses=curve_series["loss"],
            episode_metric1s=curve_series["metric1"],
            episode_metric2s=curve_series["metric2"],
            episode_fusion_counts=curve_series["fusion"],
            episode_avg_ks=curve_series["avg_k"],
            baselines={
                "loss": float(robust_reference.loss_mean),
                "metric1": float(robust_reference.metric1_mean),
                "metric2": float(robust_reference.metric2_mean),
                "avg_k": 13.0,
            },
            entropy_series=curve_series["entropy"],
            entropy_episodes=curve_series["entropy_episode"],
            metric1_name="metric1",
            metric2_name="metric2",
            log_fn=log,
        )

    best_full_vector = summary.get("best_full_vector")
    best_action_matrix = summary.get("best_action_matrix")
    result_status = _stage2_selection_result_status(summary)
    selection_completed = result_status == "completed"
    best_layer_configurations = (
        summary.get("best_layer_configurations")
        if summary.get("best_layer_configurations") is not None
        else (
            describe_layerwise_action_matrix(best_action_matrix)
            if best_action_matrix else None
        )
    )
    best_action_group = build_reloadable_best_group({
        "full_vector": best_full_vector,
        "action_matrix": best_action_matrix,
        "boosted_overrides": summary.get("best_boosted_overrides") or {},
    }) if best_full_vector is not None else None
    best_reward = summary.get("best_reward")
    if best_reward is None:
        best_reward = -float("inf")
    limits = {
        "loss": float(robust_reference.loss_limit),
        "metric1": float(robust_reference.metric1_limit),
        "metric2": float(robust_reference.metric2_limit),
        "loss_std": float(robust_reference.loss_std_limit),
        "metric1_std": float(robust_reference.metric1_std_limit),
        "metric2_std": float(robust_reference.metric2_std_limit),
    }
    return {
        "dataset_protocol_hash": getattr(
            evaluator, "dataset_protocol_hash", None
        ),
        "fixed_gelu": fixed_gelu.copy(),
        "fixed_softmax": fixed_softmax.copy(),
        "limit_loss": limits["loss"],
        "limit_p": limits["metric1"],
        "limit_s": limits["metric2"],
        "proxy_limit_loss": limits["loss"],
        "proxy_limit_p": limits["metric1"],
        "proxy_limit_s": limits["metric2"],
        "search_limits": limits,
        "all_max_blb_baseline_metrics": dict(baseline_preflight_metrics),
        "status": result_status,
        "termination_reason": completion_status,
        "scientific_status": (
            "strict_selected" if selection_completed else "no_strict_selection"
        ),
        "blb_v3_best_action_vec": best_full_vector,
        "blb_v3_best_action_group": best_action_group,
        "blb_v3_layerwise_best_action_group": best_action_group,
        "blb_v3_layerwise_best_configuration": best_layer_configurations,
        "blb_v3_best_reward": float(best_reward),
        "blb_v3_profile": str(train_cfg.profile),
        "blb_v3_fusion_count_action": True,
        "blb_v3_total_episodes": int(summary.get("completed_episodes", 0)),
        "rl_variant": rl_variant,
        "policy_network_variant": policy_network_id,
        "policy_network": policy_network_summary,
        "algorithm_revision": algorithm_revision,
        "algorithm_contract_hash": algorithm_contract_hash,
        "run_context_hash": run_context_hash,
        "communication_importance_ratio": float(
            layerwise_env.communication_importance_ratio
        ),
        "network_axis_weights": list(
            algorithm_contract["network_axis_weights"]
        ),
        "axis_precision_tolerances": list(
            algorithm_contract["axis_precision_tolerances"]
        ),
        "selection_diagnostics": {
            "selection_mode": "layerwise_network_weighted_strict",
            "best_action_vec": best_full_vector,
            "best_action_matrix": best_action_matrix,
            "best_layer_configurations": best_layer_configurations,
            "best_assessment": summary.get("best_assessment"),
            "best_metrics": summary.get("best_metrics"),
            "best_resource_objective": summary.get("best_resource_objective"),
            "strict_pareto_frontier": summary.get(
                "strict_pareto_frontier", []
            ),

            "best_variable_cost": summary.get("best_variable_cost"),
            "best_promotion_evidence": summary.get("best_promotion_evidence"),
            "best_axis_counterfactuals": summary.get(
                "best_axis_counterfactuals"
            ),
            "final_evidence": compact_summary["final_evidence"],
        },
        "sequential_diagnostics": {
            "horizon": layerwise_horizon,
            "max_step_dim": len(LAYERWISE_SLOT_NAMES),
            "state_dim": int(layerwise_env.state_dim),
            "episode_count": int(summary.get(
                "completed_episodes", completed_episode_count,
            )),
            "ppo_metric_count": int(ppo_update_counter),
            "block4_entropy": summary.get("block4_entropy"),
            "k_entropy": summary.get("k_entropy"),
            "precision_preset_entropy": summary.get("k_entropy"),
            "strict_revalidation_status": str(
                summary.get("strict_revalidation_status", "not_due")
            ),
            "termination_reason": str(
                summary.get("termination_reason") or completion_status
            ),
        },
        "layerwise_summary": summary,
    }


def _build_search_invocation_contract(
        *,
        runner: Any,
        train_cfg: Any,
        fixed_gelu: Any,
        fixed_softmax: Any,
        fixed_label: Any,
        fixed_source: Any,
        ) -> dict[str, Any]:
    from rfr.preparation.data.protocol import (
        PROTOCOL_SCHEMA as DATASET_PROTOCOL_SCHEMA,
        validate_dataset_protocol_binding,
    )
    from rfr.common.json_utils import to_jsonable
    from rfr.search.common.best_config import load_stage1_best_config

    from rfr.search.comparators.common.stage2_core import normalize_search_backend

    backend = normalize_search_backend(
        getattr(train_cfg, "search_backend", "ppo")
    )
    if backend == "ppo":
        raise ValueError("search invocation contract requires a non-PPO backend")
    evaluator = runner.evaluator
    raw_config_path = str(
        getattr(evaluator, "stage1_best_config_input_path", "") or ""
    )
    if not raw_config_path:
        raise RuntimeError("Stage-2 comparator invocation has no Stage-1 JSON")
    try:
        stage1_config_path = os.path.abspath(
            os.path.expanduser(os.fspath(raw_config_path))
        )
        stage1_config = load_stage1_best_config(stage1_config_path)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError(
            "Stage-2 comparator invocation cannot load the Stage-1 JSON"
        ) from exc

    gelu_degrees = [
        int(value) for value in np.asarray(fixed_gelu).reshape(-1)
    ]
    softmax_degrees = [
        int(value) for value in np.asarray(fixed_softmax).reshape(-1)
    ]
    num_layers = int(getattr(evaluator, "total_layers", 0))
    seed = int(getattr(train_cfg, "seed", 0))
    dataset_protocol_hash = str(
        getattr(evaluator, "dataset_protocol_hash", "") or ""
    )
    config_protocol_hash = str(
        stage1_config["provenance"].get("dataset_protocol_hash") or ""
    )
    validate_dataset_protocol_binding(
        {
            "dataset_protocol_schema": DATASET_PROTOCOL_SCHEMA,
            "dataset_protocol_hash": config_protocol_hash,
        },
        expected_hash=dataset_protocol_hash,
        artifact="Stage-2 comparator invocation",
    )
    expected_model = "bert-large" if num_layers == 24 else "bert-base"
    expected_source = f"stage1_json:{stage1_config_path}"
    if (
            str(fixed_source) != expected_source
            or stage1_config["algorithm"] != backend
            or stage1_config["model_type"] != expected_model
            or stage1_config["dataset"] != str(evaluator.dataset_key)
            or num_layers <= 0
            or stage1_config["stage1"]["gelu"] != gelu_degrees
            or stage1_config["stage1"]["softmax"] != softmax_degrees
    ):
        raise RuntimeError(
            "Stage-2 comparator binding does not match its Stage-1 JSON"
        )
    normalized_binding = {
        "schema_version": stage1_config["schema_version"],
        "algorithm": stage1_config["algorithm"],
        "model_type": stage1_config["model_type"],
        "dataset": stage1_config["dataset"],
        "num_layers": stage1_config["num_layers"],
        "gelu_degrees": gelu_degrees,
        "softmax_degrees": softmax_degrees,
        "dataset_protocol_hash": config_protocol_hash,
        "config_path": stage1_config_path,
        "config_sha256": str(
            getattr(evaluator, "stage1_best_config_input_sha256", "") or ""
        ),
    }

    scientific_parameters = {
        name: getattr(train_cfg, name, default)
        for name, default in (
            ("search_initial_design_size", 8),
            ("search_candidate_pool_size", 512),
            ("search_population_size", 24),
            ("search_bo_no_improvement_patience", 2_000),
            ("search_greedy_no_improvement_rounds", 1),
            ("search_ga_generations", 200),
            ("search_mutation_max_coordinates", 3),
            ("search_rf_n_estimators", 128),
            ("search_rf_min_samples_leaf", 2),
            ("online_num_trials_per_step", 3),
            ("baseline_groups", 5),
            ("baseline_trials_per_group", 3),
            ("promotion_validation_trials", 15),
            ("final_selection_validation_trials", 15),
            ("final_selection_top_n", 20),
            ("constraint_bootstrap_samples", 4096),
            ("online_constraint_probability", 0.50),
            ("promotion_constraint_probability", 0.80),
            ("final_constraint_probability", 0.95),
            ("stage2_stability_multiplier", 2.0),
            ("communication_importance_ratio", 1.0),
            (
                "stage1_inference_batch_size",
                getattr(evaluator, "batch_size", 1),
            ),
            (
                "stage2_inference_batch_size",
                getattr(evaluator, "batch_size", 1),
            ),
            (
                "stage2_probe_size",
                getattr(evaluator, "stage2_probe_size", 256),
            ),
            (
                "stage1_accuracy_tolerance",
                getattr(evaluator, "error_threshold", 0.001),
            ),
            (
                "stage2_limit_tolerance",
                getattr(evaluator, "stage2_limit_tolerance", 0.001),
            ),
            (
                "stage2_stability_tolerance",
                getattr(evaluator, "stage2_stability_tolerance", 1.2),
            ),
            (
                "calibrate_baseline_samples",
                getattr(train_cfg, "calibrate_baseline_samples", 8),
            ),
            (
                "truncation_backend",
                getattr(train_cfg, "truncation_backend", "binary"),
            ),
            (
                "truncation_ring_bits",
                getattr(train_cfg, "truncation_ring_bits", 43),
            ),
            (
                "truncation_source_fractional_bits",
                getattr(train_cfg, "truncation_source_fractional_bits", 24),
            ),
            (
                "terminal_eval_batch_size",
                getattr(train_cfg, "terminal_eval_batch_size", 4),
            ),
            ("decision_granularity", "layer"),
            ("reward_design", "robust_constrained"),
            ("fusion_count_action", True),
        )
    }
    return to_jsonable({
        "schema_version": "stage2_search_train_probe_invocation_v1",
        "dataset_protocol_schema": DATASET_PROTOCOL_SCHEMA,
        "dataset_protocol_hash": dataset_protocol_hash,
        "search_backend": backend,
        "profile": str(getattr(train_cfg, "profile", "")),
        "model_type": str(getattr(evaluator, "model_type", "") or ""),
        "num_layers": num_layers,
        "fixed_gelu": gelu_degrees,
        "fixed_softmax": softmax_degrees,
        "fixed_label": str(fixed_label),
        "fixed_source": str(fixed_source),
        "seed": seed,
        "stage1_selection_binding": normalized_binding,
        "stage1_config_path": stage1_config_path,
        "rng_contract": {
            "online_stream": "ppo_global_evaluation_index_v1",
            "probe_seed": "derive_layerwise_episode_probe_seed_v1",
        },
        "scientific_parameters": scientific_parameters,
    }, stringify_unknown=True)


def _selected_action_identity_payload(
        evaluation: Any,
        ) -> dict[str, Any]:
    from rfr.search.comparators.common.stage2_runner import (
        _selected_action_identity_payload as canonical_selected_identity,
    )

    return canonical_selected_identity(evaluation)


def _validate_completed_search_resume_result(
        result: Mapping[str, Any],
        expected_result: Mapping[str, Any],
        ) -> None:
    from rfr.common.json_utils import to_jsonable

    actual = to_jsonable(result, stringify_unknown=True)
    expected = to_jsonable(expected_result, stringify_unknown=True)
    if actual != expected:
        raise RuntimeError(
            "Stage-2 layerwise resume does not match the completed inner search"
        )


def _build_completed_search_resume_result(
        *,
        runner: Any,
        fixed_gelu: Any,
        fixed_softmax: Any,
        invocation_contract: Mapping[str, Any],
        inner_run: Mapping[str, Any],
        ) -> dict[str, Any]:
    from rfr.preparation.fusion.fixed_action import build_fusion_fixed_config
    from rfr.search.common.layerwise_action import describe_layerwise_action_matrix
    manifest = dict(inner_run.get("manifest") or {})
    selected = inner_run.get("selected")
    result = inner_run.get("result")
    if selected is None or result is None:
        raise RuntimeError(
            "Stage-2 completed inner search has no selected result"
        )

    resume_contract = manifest.get("resume_contract")
    requested_manifest = (
        resume_contract.get("requested_manifest")
        if isinstance(resume_contract, Mapping) else None
    )
    if (
            not isinstance(requested_manifest, Mapping)
            or requested_manifest.get("stage2_invocation")
            != dict(invocation_contract)
    ):
        raise RuntimeError(
            "Stage-2 completed inner search does not match its invocation"
        )

    search_backend = str(manifest.get("search_backend") or "")
    stage1_binding = dict(
        invocation_contract.get("stage1_selection_binding") or {}
    )
    fixed_gelu_values = [
        int(value) for value in np.asarray(fixed_gelu).reshape(-1)
    ]
    fixed_softmax_values = [
        int(value) for value in np.asarray(fixed_softmax).reshape(-1)
    ]
    if (
            search_backend != str(invocation_contract.get("search_backend") or "")
            or manifest.get("profile") != invocation_contract.get("profile")
            or int(manifest.get("num_layers", 0))
            != int(invocation_contract.get("num_layers", 0))
            or list(manifest.get("fixed_gelu") or []) != fixed_gelu_values
            or list(manifest.get("fixed_softmax") or []) != fixed_softmax_values
            or manifest.get("stage1_backend") != search_backend
            or manifest.get("stage1_bound_into_stage2") is not True
            or dict(manifest.get("stage1_selection_binding") or {})
            != stage1_binding
    ):
        raise RuntimeError(
            "Stage-2 completed inner search scientific configuration changed"
        )

    selected_metadata = dict(getattr(selected, "metadata", {}) or {})
    best_full_vector = [
        int(value)
        for value in selected_metadata.get("pending_full_vector", ())
    ]
    best_action_matrix = [
        [int(value) for value in row]
        for row in getattr(selected, "action_matrix", ())
    ]
    selected_action_identity = _selected_action_identity_payload(selected)
    if (
            selected_action_identity.get("action_matrix") != best_action_matrix
            or selected_action_identity.get("full_vector") != best_full_vector
    ):
        raise RuntimeError(
            "Stage-2 completed selected action evidence is inconsistent"
        )

    inner_status = str(manifest.get("status") or "")
    strict_feasible = bool(inner_run.get("strict_feasible", False))
    expected_status = (
        "complete_strict_feasible"
        if strict_feasible else "complete_least_violating"
    )
    if (
            inner_status != expected_status
            or inner_run.get("strict_validation") is None
    ):
        raise RuntimeError(
            "Stage-2 completed strict result has inconsistent status"
        )

    best_layer_configurations = describe_layerwise_action_matrix(
        best_action_matrix
    )
    limits = selected.limits.as_dict()
    profile = str(invocation_contract.get("profile") or "")
    num_layers = int(invocation_contract.get("num_layers", 0))
    artifact_paths = dict(inner_run.get("artifact_paths") or {})
    strict_validation = inner_run.get("strict_validation")
    scientific_parameters = invocation_contract.get("scientific_parameters")
    if not isinstance(scientific_parameters, Mapping):
        raise RuntimeError(
            "Stage-2 completed inner search has no scientific parameters"
        )
    try:
        stage2_inference_batch_size = int(
            scientific_parameters["stage2_inference_batch_size"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Stage-2 completed inner search has no valid inference batch size"
        ) from exc
    if stage2_inference_batch_size <= 0:
        raise RuntimeError(
            "Stage-2 completed inner search has no valid inference batch size"
        )
    common = {
        "dataset_protocol_hash": manifest.get("dataset_protocol_hash"),
        "fixed_gelu": np.asarray(fixed_gelu, dtype=int).copy(),
        "fixed_softmax": np.asarray(fixed_softmax, dtype=int).copy(),
        "strict_feasible": strict_feasible,
        "stage2_inference_batch_size": stage2_inference_batch_size,
        "selected_action_identity": selected_action_identity,
        "search_backend": search_backend,
        "stage1_consumed_binding": stage1_binding,
        "strict_identity_context_hash": str(
            manifest.get("strict_identity_context_hash") or ""
        ),
        "final_config_fingerprint": selected_action_identity[
            "final_config_fingerprint"
        ],
        "blb_v3_profile": profile,
        "blb_v3_total_episodes": int(result.evaluation_count),
        "limit_loss": float(limits["loss_max"]),
        "limit_p": float(limits["metric1_min"]),
        "limit_s": float(limits["metric2_min"]),
    }

    fixed_config = build_fusion_fixed_config(
        best_full_vector,
        profile=profile,
        num_layers=num_layers,
        gelu=np.asarray(fixed_gelu, dtype=int),
        softmax=np.asarray(fixed_softmax, dtype=int),
        source=f"stage2_{search_backend}_best",
    )
    best_action_group = dict(fixed_config["group"])
    best_action_group["policy_actions"] = best_action_matrix
    best_action_group["boosted_overrides"] = selected_metadata.get(
        "boosted_overrides", []
    )
    accounting_names = (
        "seed",
        "observation_count",
        "inference_reaching_candidate_count",
        "online_candidate_trial_count",
        "strict_evaluated_candidate_count",
        "strict_joint_trial_count",
        "strict_compute_trial_count",
        "strict_communication_trial_count",
        "strict_total_evidence_trial_count",
        "strict_fresh_trial_count",
        "total_candidate_trial_count",
        "model_forward_trial_count",
        "online_search_wall_seconds",
        "strict_attempt_count",
        "strict_attempt_wall_seconds_total",
        "strict_validation_wall_seconds",
        "total_wall_seconds",
        "termination_reason",
    )
    return {
        **common,
        "status": "completed" if strict_feasible else "completed_infeasible",
        "scientific_status": (
            "full_search_with_strict_train_probe_gate"
            if strict_feasible else "full_search_strict_least_violating"
        ),
        "search_accounting": {
            key: manifest.get(key) for key in accounting_names
        },
        "rl_variant": f"blb_v3_layerwise_search_{search_backend}",
        "blb_v3_best_action_vec": best_full_vector,
        "blb_v3_best_action_group": best_action_group,
        "blb_v3_layerwise_best_action_group": best_action_group,
        "blb_v3_layerwise_best_configuration": best_layer_configurations,
        "blb_v3_best_reward": float(selected.reward or 0.0),
        "blb_v3_fusion_count_action": True,
        "proxy_limit_loss": float(limits["loss_max"]),
        "proxy_limit_p": float(limits["metric1_min"]),
        "proxy_limit_s": float(limits["metric2_min"]),
        "search_limits": {
            "loss": float(limits["loss_max"]),
            "metric1": float(limits["metric1_min"]),
            "metric2": float(limits["metric2_min"]),
            "loss_std": float(limits["loss_std_max"]),
            "metric1_std": float(limits["metric1_std_max"]),
            "metric2_std": float(limits["metric2_std_max"]),
        },
        "selection_diagnostics": {
            "selection_mode": "layerwise_constrained_search_baseline",
            "best_action_matrix": best_action_matrix,
            "best_layer_configurations": best_layer_configurations,
            "best_evaluation": selected.as_dict(),
            "strict_validation": strict_validation,
            "artifact_paths": artifact_paths,
        },
        "layerwise_summary": {
            **result.as_dict(),
            "selected": selected.as_dict(),
            "strict_validation": strict_validation,
            "artifact_paths": artifact_paths,
        },
    }


def _restore_search_resume_result(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise RuntimeError("Stage-2 layerwise resume result is not an object")
    restored = dict(payload)
    for name in ("fixed_gelu", "fixed_softmax"):
        if name in restored:
            restored[name] = np.asarray(restored[name], dtype=int)
    return restored


def _write_completed_search_resume(
        *,
        search_output_dir: str,
        invocation_contract: Mapping[str, Any],
        result: Mapping[str, Any],
        expected_result: Mapping[str, Any],
        ) -> None:
    from rfr.common.json_utils import read_json_file

    from rfr.search.comparators.common.stage2_runner import _atomic_json

    invocation_path = os.path.join(search_output_dir, "invocation.json")
    if (
            not os.path.isfile(invocation_path)
            or read_json_file(invocation_path) != dict(invocation_contract)
    ):
        raise RuntimeError(
            "Stage-2 completed search invocation does not match invocation.json"
        )
    _validate_completed_search_resume_result(result, expected_result)

    resume_result_path = os.path.join(search_output_dir, "resume_result.json")
    if os.path.isfile(resume_result_path):
        restored = _restore_search_resume_result(
            read_json_file(resume_result_path)
        )
        _validate_completed_search_resume_result(restored, expected_result)
        _validate_completed_search_resume_result(result, restored)
        return
    _atomic_json(resume_result_path, result)


_EPISODE_METRICS_RESUME_FIELDS = (
    "loss_mean",
    "loss_std",
    "metric1_mean",
    "metric2_mean",
    "metric1_std",
    "metric2_std",
    "loss_max",
    "metric1_min",
    "metric2_min",
    "loss_trials",
    "metric1_trials",
    "metric2_trials",
    "trial_seeds",
)


def _episode_metrics_resume_payload(metrics: Any) -> dict[str, Any]:
    """Serialize the exact clean-probe evidence needed for zero-forward resume."""
    payload = {
        "schema_version": "stage2_episode_metrics_resume_v1",
    }
    for name in _EPISODE_METRICS_RESUME_FIELDS[:9]:
        value = float(getattr(metrics, name))
        if not math.isfinite(value):
            raise ValueError(f"clean baseline metric {name} must be finite")
        payload[name] = value
    for name in _EPISODE_METRICS_RESUME_FIELDS[9:12]:
        values = tuple(float(value) for value in getattr(metrics, name))
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"clean baseline metric trial series {name} must be finite and non-empty"
            )
        payload[name] = list(values)
    trial_seeds = tuple(int(value) for value in metrics.trial_seeds)
    trial_count = len(payload["loss_trials"])
    if (
            len(payload["metric1_trials"]) != trial_count
            or len(payload["metric2_trials"]) != trial_count
            or len(trial_seeds) not in {0, trial_count}
    ):
        raise ValueError("clean baseline metric trial series lengths must match")
    payload["trial_seeds"] = list(trial_seeds)
    return payload


def _episode_metrics_from_resume_payload(payload: Mapping[str, Any]) -> Any:
    """Authenticate and restore one clean-probe EpisodeMetrics payload."""
    from rfr.search.rl.stage2.reward import EpisodeMetrics

    required = {"schema_version", *_EPISODE_METRICS_RESUME_FIELDS}
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("clean baseline metrics resume payload is invalid")
    if payload.get("schema_version") != "stage2_episode_metrics_resume_v1":
        raise ValueError("clean baseline metrics resume schema mismatch")
    restored_values = {
        name: payload[name]
        for name in _EPISODE_METRICS_RESUME_FIELDS[:9]
    }
    restored_values.update({
        name: tuple(payload[name])
        for name in _EPISODE_METRICS_RESUME_FIELDS[9:]
    })
    restored = EpisodeMetrics(**restored_values)
    if _episode_metrics_resume_payload(restored) != dict(payload):
        raise ValueError("clean baseline metrics resume payload changed on restore")
    return restored


def _restore_pending_strict_resume_evidence(
        context: Mapping[str, Any],
        *,
        precision_tolerance: float,
        stability_multiplier: float,
        bootstrap_samples: int,
        online_trials: int,
        baseline_groups: int,
        trials_per_group: int,
        authoritative_example_count: int,
        ) -> dict[str, Any]:
    """Restore and validate every baseline reference used by strict resume."""
    from rfr.search.rl.stage2.layerwise_runner import LayerwiseValidationBanks
    from rfr.search.common.statistical_constraints import baseline_reference_from_resume_payload

    if not isinstance(context, Mapping):
        raise ValueError("pending strict resume context must be a mapping")
    clean_metrics = _episode_metrics_from_resume_payload(
        context.get("clean_baseline_metrics")
    )
    if len(tuple(clean_metrics.loss_trials)) != int(online_trials):
        raise RuntimeError(
            "pending strict clean baseline trial count changed"
        )
    robust_reference = baseline_reference_from_resume_payload(
        context.get("robust_reference")
    )
    validation_banks = LayerwiseValidationBanks.from_resume_payload(
        context.get("validation_banks")
    )
    expected_trials = int(baseline_groups) * int(trials_per_group)
    expected_reference_contract = (
        float(precision_tolerance),
        float(stability_multiplier),
        int(bootstrap_samples),
    )

    def validate_reference(reference: Any, trial_count: int, label: str) -> None:
        actual_contract = (
            float(reference.precision_tolerance),
            float(reference.stability_multiplier),
            int(reference.bootstrap_samples),
        )
        if actual_contract != expected_reference_contract:
            raise RuntimeError(
                f"pending strict {label} baseline contract changed"
            )
        if int(reference.trial_count) != int(trial_count):
            raise RuntimeError(
                f"pending strict {label} baseline trial count changed"
            )

    validate_reference(robust_reference, expected_trials, "online")
    for bank in (
            validation_banks.bank_a,
            validation_banks.bank_b,
            validation_banks.bank_c,
    ):
        if (
                len(tuple(bank.probe_seeds)) != int(baseline_groups)
                or int(bank.trials_per_probe) != int(trials_per_group)
                or int(bank.trial_count) != expected_trials
        ):
            raise RuntimeError(
                f"pending strict validation Bank {bank.label} contract changed"
            )
        validate_reference(
            bank.reference, expected_trials, f"Bank {bank.label}"
        )
    validate_reference(
        validation_banks.promotion_reference,
        2 * expected_trials,
        "promotion A+B",
    )
    validate_reference(
        validation_banks.final_reference,
        3 * expected_trials,
        "final A+B+C",
    )

    stored_example_count = int(
        context.get("authoritative_validation_example_count", 0)
    )
    if (
            int(authoritative_example_count) != 408
            or stored_example_count != int(authoritative_example_count)
    ):
        raise RuntimeError(
            "pending strict authoritative validation example count changed"
        )
    baseline_preflight_metrics = context.get("baseline_preflight_metrics")
    authoritative_summary = context.get("authoritative_robust_summary")
    if not isinstance(baseline_preflight_metrics, Mapping):
        raise ValueError("pending strict baseline preflight metrics are invalid")
    if not isinstance(authoritative_summary, Mapping) or not authoritative_summary:
        raise ValueError("pending strict authoritative baseline summary is invalid")
    return {
        "clean_baseline_metrics": clean_metrics,
        "robust_reference": robust_reference,
        "baseline_preflight_metrics": dict(baseline_preflight_metrics),
        "validation_banks": validation_banks,
        "authoritative_robust_summary": dict(authoritative_summary),
        "authoritative_validation_example_count": stored_example_count,
    }


def _write_pending_strict_resume_context(
        *,
        search_output_dir: str,
        invocation_contract: Mapping[str, Any],
        resume_contract: Mapping[str, Any],
        clean_baseline_metrics: Mapping[str, Any],
        robust_reference: Mapping[str, Any],
        baseline_preflight_metrics: Mapping[str, Any],
        validation_banks: Mapping[str, Any],
        authoritative_robust_summary: Mapping[str, Any],
        authoritative_validation_example_count: int,
        ) -> None:
    """Persist the baseline evidence needed to restart strict validation."""
    from rfr.common.json_utils import read_json_file

    from rfr.search.comparators.common.stage2_runner import _atomic_json

    payload = {
        "schema_version": "stage2_pending_strict_resume_context_v2",
        "invocation_contract": dict(invocation_contract),
        "resume_contract": dict(resume_contract),
        "clean_baseline_metrics": dict(clean_baseline_metrics),
        "robust_reference": dict(robust_reference),
        "baseline_preflight_metrics": dict(baseline_preflight_metrics),
        "validation_banks": dict(validation_banks),
        "authoritative_robust_summary": dict(
            authoritative_robust_summary
        ),
        "authoritative_validation_example_count": int(
            authoritative_validation_example_count
        ),
    }
    os.makedirs(search_output_dir, exist_ok=True)
    path = os.path.join(
        search_output_dir, "pending_strict_resume_context.json",
    )
    if os.path.isfile(path):
        if read_json_file(path) != payload:
            raise RuntimeError(
                "Stage-2 pending strict resume context changed"
            )
        return
    _atomic_json(path, payload)


def _preflight_pending_strict_search_resume(
        *,
        runner: Any,
        train_cfg: Any,
        fixed_gelu: Any,
        fixed_softmax: Any,
        fixed_label: Any,
        fixed_source: Any,
        blb_progress_dir: str,
        ) -> dict[str, Any] | None:
    """Restore ordinary pending-strict evidence before model setup."""
    from rfr.preparation.data.protocol import validate_dataset_protocol_binding
    from rfr.common.json_utils import read_json_file

    from rfr.search.comparators.common.stage2_core import normalize_search_backend

    backend = normalize_search_backend(
        getattr(train_cfg, "search_backend", "ppo")
    )
    if backend == "ppo":
        return None
    search_output_dir = os.path.join(
        blb_progress_dir, f"search_{backend}",
    )
    manifest_path = os.path.join(search_output_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        return None

    manifest = read_json_file(manifest_path)
    if not isinstance(manifest, Mapping):
        raise RuntimeError("Stage-2 pending strict manifest is invalid")
    validate_dataset_protocol_binding(
        manifest,
        expected_hash=getattr(runner.evaluator, "dataset_protocol_hash", None),
        artifact="Stage-2 pending strict manifest",
    )
    status = str(manifest.get("status") or "")
    if status in {
            "complete_strict_feasible",
            "complete_least_violating",
    }:
        return None
    if status != "search_complete_pending_strict":
        return None

    invocation = _build_search_invocation_contract(
        runner=runner,
        train_cfg=train_cfg,
        fixed_gelu=fixed_gelu,
        fixed_softmax=fixed_softmax,
        fixed_label=fixed_label,
        fixed_source=fixed_source,
    )
    invocation_path = os.path.join(search_output_dir, "invocation.json")
    if (
            not os.path.isfile(invocation_path)
            or read_json_file(invocation_path) != invocation
    ):
        raise RuntimeError(
            "Stage-2 pending strict invocation does not match existing artifacts"
        )

    context_path = os.path.join(
        search_output_dir, "pending_strict_resume_context.json",
    )
    if not os.path.isfile(context_path):
        raise RuntimeError(
            "Stage-2 pending strict run has no resume context"
        )
    context = read_json_file(context_path)
    required_fields = {
        "schema_version",
        "invocation_contract",
        "resume_contract",
        "clean_baseline_metrics",
        "robust_reference",
        "baseline_preflight_metrics",
        "validation_banks",
        "authoritative_robust_summary",
        "authoritative_validation_example_count",
    }
    if not isinstance(context, Mapping) or set(context) != required_fields:
        raise RuntimeError("Stage-2 pending strict resume context is invalid")
    context = dict(context)
    resume_contract = context.get("resume_contract")
    requested_manifest = (
        resume_contract.get("requested_manifest")
        if isinstance(resume_contract, Mapping) else None
    )
    if (
            context.get("schema_version")
            != "stage2_pending_strict_resume_context_v2"
            or context.get("invocation_contract") != invocation
            or not isinstance(resume_contract, Mapping)
            or manifest.get("resume_contract") != resume_contract
            or not isinstance(requested_manifest, Mapping)
            or requested_manifest.get("stage2_invocation") != invocation
    ):
        raise RuntimeError(
            "Stage-2 pending strict resume context does not match the run"
        )
    return context


def _preflight_completed_search_resume(
        *,
        runner: Any,
        train_cfg: Any,
        fixed_gelu: Any,
        fixed_softmax: Any,
        fixed_label: Any,
        fixed_source: Any,
        blb_progress_dir: str,
        ) -> dict[str, Any] | None:
    from rfr.preparation.data.protocol import validate_dataset_protocol_binding
    from rfr.common.json_utils import read_json_file

    from rfr.search.comparators.common.stage2_runner import (
        _atomic_json,
        _load_plain_completed_search_run,
    )
    from rfr.search.comparators.common.stage2_core import normalize_search_backend

    backend = normalize_search_backend(
        getattr(train_cfg, "search_backend", "ppo")
    )
    if backend == "ppo":
        return None

    search_output_dir = os.path.join(
        blb_progress_dir, f"search_{backend}",
    )
    os.makedirs(search_output_dir, exist_ok=True)
    invocation = _build_search_invocation_contract(
        runner=runner,
        train_cfg=train_cfg,
        fixed_gelu=fixed_gelu,
        fixed_softmax=fixed_softmax,
        fixed_label=fixed_label,
        fixed_source=fixed_source,
    )
    invocation_path = os.path.join(search_output_dir, "invocation.json")
    resume_result_path = os.path.join(search_output_dir, "resume_result.json")
    persisted_invocation = None
    if os.path.isfile(invocation_path):
        persisted_invocation = read_json_file(invocation_path)
        if not isinstance(persisted_invocation, Mapping):
            raise RuntimeError("Stage-2 comparator invocation is invalid")
    else:
        existing_names = {
            name for name in os.listdir(search_output_dir)
            if name not in {"invocation.json.tmp"}
        }
        if existing_names:
            raise RuntimeError(
                "Stage-2 search artifacts exist without invocation.json"
            )
        _atomic_json(invocation_path, invocation)
        persisted_invocation = invocation

    manifest_path = os.path.join(search_output_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        if persisted_invocation != invocation:
            raise RuntimeError(
                "Stage-2 comparator invocation does not match existing artifacts"
            )
        if os.path.isfile(resume_result_path):
            raise RuntimeError(
                "Stage-2 resume result exists without an inner manifest"
            )
        return None
    manifest = read_json_file(manifest_path)
    if not isinstance(manifest, Mapping):
        raise RuntimeError("Stage-2 inner search manifest is invalid")
    validate_dataset_protocol_binding(
        manifest,
        expected_hash=getattr(runner.evaluator, "dataset_protocol_hash", None),
        artifact="Stage-2 completed search manifest",
    )
    status = str(manifest.get("status") or "")
    completed_statuses = {
        "complete_strict_feasible",
        "complete_least_violating",
    }
    resume_contract = manifest.get("resume_contract")
    requested_manifest = (
        resume_contract.get("requested_manifest")
        if isinstance(resume_contract, Mapping) else None
    )
    if persisted_invocation != invocation:
        raise RuntimeError(
            "Stage-2 comparator invocation does not match existing artifacts"
        )
    if status not in completed_statuses:
        if os.path.isfile(resume_result_path):
            raise RuntimeError(
                "Stage-2 resume result exists before inner search completion"
            )
        return None

    if (
            not isinstance(requested_manifest, Mapping)
            or requested_manifest.get("stage2_invocation") != invocation
    ):
        raise RuntimeError(
            "Stage-2 completed inner search invocation does not match"
        )

    inner_run = _load_plain_completed_search_run(
        output_dir=search_output_dir,
        manifest=manifest,
        communication_importance_ratio=float(
            manifest.get("communication_importance_ratio", 1.0)
        ),
    )
    expected_result = _build_completed_search_resume_result(
        runner=runner,
        fixed_gelu=fixed_gelu,
        fixed_softmax=fixed_softmax,
        invocation_contract=invocation,
        inner_run=inner_run,
    )
    if os.path.isfile(resume_result_path):
        restored = _restore_search_resume_result(
            read_json_file(resume_result_path)
        )
        _validate_completed_search_resume_result(
            restored, expected_result,
        )
        return restored

    _write_completed_search_resume(
        search_output_dir=search_output_dir,
        invocation_contract=invocation,
        result=expected_result,
        expected_result=expected_result,
    )
    return expected_result


def _build_stage2_materialization_env(
        *,
        runner: Any,
        train_cfg: Any,
        fixed_gelu: Any,
        fixed_softmax: Any,
        log: Callable[[str], None],
        ) -> dict[str, Any]:
    from rfr.preparation.rescale.baseline_bootstrap import (
        load_calibrated_stage2_action_context,
        validate_calibrated_stage2_action_context,
    )
    from rfr.search.rl.stage2.env import BLBStage2Env, BLBStage2EnvConfig
    from rfr.search.rl.stage2.reward import BaselineCostStats, RewardWeights

    ev = runner.evaluator
    gelu = np.asarray(fixed_gelu, dtype=int)
    softmax = np.asarray(fixed_softmax, dtype=int)
    ev.apply_configuration(gelu, softmax)

    probe_batches = runner._build_probe_batches(
        ev,
        train_cfg,
        probe_size_override=256,
    )
    train_cfg.probe_batch_count = max(
        1, int(len(probe_batches) or train_cfg.probe_batch_count)
    )
    log(f"  * 评估子集: batch 数 = {len(probe_batches)}")
    rescale_bridge = runner._build_rescale_bridge(train_cfg, log=log)
    calibrated_action_context = load_calibrated_stage2_action_context(
        rescale_optimizer_root=str(train_cfg.inproc_rescale_optimizer_root),
        dataset=str(train_cfg.profile),
        num_layers=int(ev.total_layers),
        gelu_per_layer=[int(value) for value in gelu.reshape(-1)],
        softmax_per_layer=[int(value) for value in softmax.reshape(-1)],
        snap_sf_to_noise_table=False,
    )
    validate_calibrated_stage2_action_context(
        calibrated_action_context,
        dataset=str(train_cfg.profile),
        num_layers=int(ev.total_layers),
        gelu_per_layer=[int(value) for value in gelu.reshape(-1)],
        softmax_per_layer=[int(value) for value in softmax.reshape(-1)],
        snap_sf_to_noise_table=False,
    )
    baseline_object = calibrated_action_context.baseline
    baseline_action_vec = np.asarray(
        calibrated_action_context.baseline_action_vec,
        dtype=np.int64,
    ).reshape(-1)
    log(
        "  * calibrated static_skeletons baseline loaded from "
        f"{baseline_object.archive_path} "
        "(sha256="
        f"{calibrated_action_context.provenance['archive_sha256']})"
    )
    base_env = BLBStage2Env(
        handler=ev.reversible_handler,
        model=ev.model,
        probe_batches=probe_batches,
        rescale_bridge=rescale_bridge,
        baseline=BaselineCostStats(),
        reward_weights=RewardWeights(),
        acc_threshold=train_cfg.acc_threshold,
        stab_threshold=train_cfg.stab_threshold,
        max_sfs=calibrated_action_context.max_sfs,
        num_layers=int(ev.total_layers),
        gelu_degree=gelu,
        attn_degree=softmax,
        layers_attribute="model." + ev.layers_attribute,
        is_regression=bool(getattr(ev, "is_regression", False)),
        env_cfg=BLBStage2EnvConfig(
            profile=train_cfg.profile,
            num_trials_per_step=train_cfg.num_trials_per_step,
            probe_batch_count=train_cfg.probe_batch_count,
            truncation_backend=train_cfg.truncation_backend,
            truncation_ring_bits=train_cfg.truncation_ring_bits,
            truncation_source_fractional_bits=(
                train_cfg.truncation_source_fractional_bits
            ),
            borderline_retest_enabled=False,
            borderline_retest_trials_multiplier=1,
        ),
    )
    base_env.pareto_cost_archive = None
    base_env.sync_degree_vectors_from_model()
    return {
        "base_env": base_env,
        "baseline_action_vec": baseline_action_vec,
        "calibrated_action_context": calibrated_action_context,
        "baseline_object": baseline_object,
        "baseline_cost_stats": calibrated_action_context.cost_stats,
    }


class _ProbeRunnerOwnerHolder:
    """Own one shared probe pool across every Stage-2 exit path."""

    def __init__(self) -> None:
        self._owner: Any | None = None
        self._closed = False

    def bind(self, owner: Any) -> None:
        if owner is None:
            raise ValueError("probe runner owner must not be None")
        if self._closed:
            raise RuntimeError("probe runner owner holder is already closed")
        if self._owner is None:
            self._owner = owner
            return
        if self._owner is not owner:
            raise RuntimeError("probe runner owner holder is already bound")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owner is not None:
            self._owner.close()


def run_layerwise_via_runner(
        *,
        runner,
        train_cfg,
        fixed_gelu,
        fixed_softmax,
        fixed_label,
        fixed_source,
        resume_checkpoint_path=None,
        ) -> Dict[str, Any]:
    """Lock the complete Stage-2 run before any probe or persistent write."""
    validate_exact_k_domain(K_LEVELS)
    from rfr.search.rl.stage2.layerwise_runner import LayerwiseRunLock
    from rfr.search.rl.stage2.training import resolve_blb_persistence_dir

    blb_progress_dir = resolve_blb_persistence_dir(runner.evaluator)
    probe_runner_owner_holder = _ProbeRunnerOwnerHolder()
    with LayerwiseRunLock(blb_progress_dir) as run_lock:
        try:
            completed_resume = _preflight_completed_search_resume(
                runner=runner,
                train_cfg=train_cfg,
                fixed_gelu=fixed_gelu,
                fixed_softmax=fixed_softmax,
                fixed_label=fixed_label,
                fixed_source=fixed_source,
                blb_progress_dir=blb_progress_dir,
            )
            if completed_resume is not None:
                return completed_resume

            pending_strict_resume_context = (
                _preflight_pending_strict_search_resume(
                    runner=runner,
                    train_cfg=train_cfg,
                    fixed_gelu=fixed_gelu,
                    fixed_softmax=fixed_softmax,
                    fixed_label=fixed_label,
                    fixed_source=fixed_source,
                    blb_progress_dir=blb_progress_dir,
                )
            )
            result = _run_layerwise_via_runner_locked(
                runner=runner,
                train_cfg=train_cfg,
                fixed_gelu=fixed_gelu,
                fixed_softmax=fixed_softmax,
                fixed_label=fixed_label,
                fixed_source=fixed_source,
                resume_checkpoint_path=resume_checkpoint_path,
                run_lock=run_lock,
                probe_runner_owner_holder=probe_runner_owner_holder,
                pending_strict_resume_context=pending_strict_resume_context,
            )
            from rfr.search.comparators.common.stage2_core import normalize_search_backend

            backend = normalize_search_backend(
                getattr(train_cfg, "search_backend", "ppo")
            )
            if backend == "ppo":
                return result

            completed_result = _preflight_completed_search_resume(
                runner=runner,
                train_cfg=train_cfg,
                fixed_gelu=fixed_gelu,
                fixed_softmax=fixed_softmax,
                fixed_label=fixed_label,
                fixed_source=fixed_source,
                blb_progress_dir=blb_progress_dir,
            )
            if completed_result is None:
                raise RuntimeError(
                    "Stage-2 comparator finished without completed inner artifacts"
                )
            _validate_completed_search_resume_result(
                result, completed_result,
            )
            return completed_result
        finally:
            probe_runner_owner_holder.close()


def _run_layerwise_via_runner_locked(
        *,
        runner,
        train_cfg,
        fixed_gelu,
        fixed_softmax,
        fixed_label,
        fixed_source,
        resume_checkpoint_path=None,
        run_lock: Any,
        probe_runner_owner_holder: _ProbeRunnerOwnerHolder,
        pending_strict_resume_context: Mapping[str, Any] | None = None,
        ) -> Dict[str, Any]:
    """Drive the layerwise RL pipeline using BLBStage2RLRunner's setup helpers.

    Reuses the runner's probe, Rescale, baseline, and persistence owners so the
    policy and strict evaluation share one materialization path.

    Returns the selected action, strict evidence, and materialization metadata.
    """
    from rfr.search.common.persistence import (
        BLBStatusBoard,
    )
    from rfr.search.runtime.probe_runner import enable_cuda_reward_probe_fast_math
    from rfr.search.rl.stage2.reward import ParetoCostArchive
    from rfr.search.rl.stage2.training import resolve_blb_persistence_dir


    enable_cuda_reward_probe_fast_math()
    ev = runner.evaluator
    robust_mode = True
    decision_path = "layerwise"
    restored_pending_evidence = None
    precision_tolerance = None
    stability_multiplier = None
    bootstrap_samples = None
    configured_baseline_groups = None
    configured_baseline_trials = None
    if pending_strict_resume_context is not None:
        if not robust_mode or decision_path != "layerwise":
            raise RuntimeError(
                "pending strict resume requires layerwise robust constrained mode"
            )
        precision_tolerance, stability_multiplier, bootstrap_samples = (
            _resolve_robust_baseline_config(train_cfg, ev)
        )
        from rfr.search.rl.stage2.layerwise_runner import (
            validate_layerwise_validation_bank_config,
        )

        configured_baseline_groups, configured_baseline_trials = (
            validate_layerwise_validation_bank_config(train_cfg)
        )
        restored_pending_evidence = _restore_pending_strict_resume_evidence(
            pending_strict_resume_context,
            precision_tolerance=precision_tolerance,
            stability_multiplier=stability_multiplier,
            bootstrap_samples=bootstrap_samples,
            online_trials=int(train_cfg.num_trials_per_step),
            baseline_groups=configured_baseline_groups,
            trials_per_group=configured_baseline_trials,
            authoritative_example_count=408,
        )
    bullet = "*"
    log = runner._make_log_safe(ev.log)
    active_rl_mode = "layerwise_robust"


    blb_progress_dir = resolve_blb_persistence_dir(ev)
    try:
        ev.noise_stage_progress_dir = blb_progress_dir
    except Exception:
        pass

    _seq_log_major_rule(
        log,
        (
            "阶段 5 · 二阶段噪声强化学习"
            f"（BLB v3 · {int(ev.total_layers)}-step layerwise robust）"
        ),
    )
    log(
        f"  {bullet} 模式（mode）："
        + f"horizon={int(ev.total_layers)} layerwise，max_step_dim=2"
    )
    log(f"  {bullet} 固定 GELU/Softmax 来源（source）：{fixed_source}    标签（label）：{fixed_label}")
    log(f"  {bullet} GELU 离散阶数向量:   {np.asarray(fixed_gelu, dtype=int).tolist()}")
    log(f"  {bullet} Softmax 离散阶数向量: {np.asarray(fixed_softmax, dtype=int).tolist()}")
    log(
        f"  {bullet} 训练概览：profile={train_cfg.profile!r}    "
        f"total_episodes={train_cfg.total_episodes}    "
        f"PPO 更新间隔（rollout_size）= {max(1, int(train_cfg.rollout_size))}    "
        f"seed={int(train_cfg.seed)}"
    )
    log(f"  {bullet} BLB 持久化目录：{blb_progress_dir}")

    run_basename = os.path.basename(os.path.normpath(str(getattr(ev, "run_output_dir", "") or "")))\
        or "blb_stage2_default_run"
    status = BLBStatusBoard(
        blb_progress_dir,
        total_episodes=int(train_cfg.total_episodes),
        profile=str(train_cfg.profile),
        run_basename=run_basename,
        extra_meta={
            "fixed_label": str(fixed_label),
            "fixed_source": str(fixed_source),
            "rl_mode": active_rl_mode,
            "rescale_optimizer": "in_process_real",
            "rescale_optimizer_root": str(train_cfg.inproc_rescale_optimizer_root),
        },
        log_fn=log,
    )
    status.set_phase("装载 stage1 GELU/Softmax 多项式近似")


    materialization_setup = _build_stage2_materialization_env(
        runner=runner,
        train_cfg=train_cfg,
        fixed_gelu=fixed_gelu,
        fixed_softmax=fixed_softmax,
        log=log,
    )
    fixed_gelu = np.asarray(fixed_gelu, dtype=int)
    fixed_softmax = np.asarray(fixed_softmax, dtype=int)
    base_env = materialization_setup["base_env"]
    calibrated_action_context = materialization_setup[
        "calibrated_action_context"
    ]
    ss_baseline_obj = materialization_setup["baseline_object"]
    ss_cost_stats = materialization_setup["baseline_cost_stats"]
    ss_action_vec = calibrated_action_context.baseline_action_vec
    max_sfs = calibrated_action_context.max_sfs
    baseline_action_vec = materialization_setup["baseline_action_vec"]


    reward_devices = list(getattr(train_cfg, "reward_devices", []) or [])
    if reward_devices and len(reward_devices) >= 2:
        from rfr.search.runtime.probe_runner import build_probe_runner
        log(f"  [multi-gpu] reward probe enabled: devices={reward_devices}")
        shared_probe_runner_owner = build_probe_runner(
            primary_model=ev.model,
            primary_handler=ev.reversible_handler,
            primary_bridge=base_env.bridge,
            primary_probe_batches=base_env.probe_batches,
            layers_attribute="model." + ev.layers_attribute,
            is_regression=bool(getattr(ev, "is_regression", False)),
            device_ids=reward_devices,
            metric_profile=str(train_cfg.profile),
            log_fn=lambda m: log(f"  [multi-gpu] {m}"),
        )
        probe_runner_owner_holder.bind(shared_probe_runner_owner)
        base_env._shared_probe_runner_owner = shared_probe_runner_owner
        base_env._shared_probe_batch_sets = {
            "F1": tuple(base_env.probe_batches),
        }
        base_env.probe_runner = shared_probe_runner_owner.view("F1")


    from rfr.search.rl.stage2.env import estimate_baseline_cost_stats
    from rfr.search.rl.stage2.reward import calibrate_weights_from_baseline
    precomputed = {
        "total_bits_sum": int(ss_cost_stats.total_bits_sum),
        "total_fusion_count": int(ss_cost_stats.total_fusion_count),
        "avg_k": float(ss_cost_stats.avg_k),
    }
    baseline = estimate_baseline_cost_stats(
        base_env,
        sample_count=int(train_cfg.calibrate_baseline_samples),
        precomputed_baseline_signals=precomputed,
    )
    base_env.baseline = baseline


    if pending_strict_resume_context is None:
        baseline_metrics = runner._estimate_baseline_metrics(base_env)
    else:
        baseline_metrics = restored_pending_evidence[
            "clean_baseline_metrics"
        ]
    baseline.loss_mean = float(baseline_metrics.loss_mean)
    baseline.loss_std = float(baseline_metrics.loss_std)
    baseline.metric1_mean = float(baseline_metrics.metric1_mean)
    baseline.metric2_mean = float(baseline_metrics.metric2_mean)


    baseline.metric1_std = float(getattr(baseline_metrics, "metric1_std", 0.0) or 0.0)
    baseline.metric2_std = float(getattr(baseline_metrics, "metric2_std", 0.0) or 0.0)
    baseline_clean_metric1 = float(baseline_metrics.metric1_mean)
    baseline_clean_metric2 = float(baseline_metrics.metric2_mean)


    baseline.typical_bits_drop = float(
        max(baseline.total_bits_sum / max(int(base_env.num_layers), 1), 1.0)
    )
    baseline.typical_fusion_count = float(base_env.num_layers)
    baseline.typical_k_drop = 5.0


    weights = calibrate_weights_from_baseline(baseline)
    base_env.reward_weights = weights


    noisy_baseline_metric1 = baseline_clean_metric1
    noisy_baseline_metric2 = baseline_clean_metric2
    noisy_baseline_loss_std = 0.0
    noisy_baseline_metric1_std = 0.0
    noisy_baseline_metric2_std = 0.0
    noisy_baseline_loss_mean = float(baseline.loss_mean)
    preflight_ok = False


    def run_standard_preflight() -> None:
        nonlocal noisy_baseline_metric1
        nonlocal noisy_baseline_metric2
        nonlocal noisy_baseline_loss_std
        nonlocal noisy_baseline_metric1_std
        nonlocal noisy_baseline_metric2_std
        nonlocal noisy_baseline_loss_mean
        nonlocal preflight_ok
        try:
            base_env.reset(seed=int(train_cfg.seed))
            _, _preflight_reward, _, preflight_info = base_env.step(baseline_action_vec)
            noisy_metrics = preflight_info.get("metrics")
            if noisy_metrics is not None:
                noisy_baseline_metric1 = float(getattr(noisy_metrics, "metric1_mean", baseline_clean_metric1))
                noisy_baseline_metric2 = float(getattr(noisy_metrics, "metric2_mean", baseline_clean_metric2))
                raw_std = float(getattr(noisy_metrics, "loss_std", 0.0))
                noisy_baseline_loss_std = raw_std if np.isfinite(raw_std) else 0.0
                raw_m1_std = float(getattr(noisy_metrics, "metric1_std", 0.0))
                noisy_baseline_metric1_std = raw_m1_std if np.isfinite(raw_m1_std) else 0.0
                raw_m2_std = float(getattr(noisy_metrics, "metric2_std", 0.0))
                noisy_baseline_metric2_std = raw_m2_std if np.isfinite(raw_m2_std) else 0.0
                raw_mean = float(getattr(noisy_metrics, "loss_mean", baseline.loss_mean))
                noisy_baseline_loss_mean = raw_mean if np.isfinite(raw_mean) else float(baseline.loss_mean)


                baseline.loss_std = noisy_baseline_loss_std
                baseline.metric1_std = noisy_baseline_metric1_std
                baseline.metric2_std = noisy_baseline_metric2_std
                preflight_ok = True
        except Exception as exc:
            log(f"  [baseline-preflight][warning] noisy probe failed: {exc}")

    _run_standard_preflight_if_needed(
        robust_mode=robust_mode,
        run_standard_preflight=run_standard_preflight,
    )


    allowed_acc_drop = max(0.0, float(getattr(ev, "stage2_limit_tolerance", 0.05)))
    stability_tol = max(0.0, float(getattr(ev, "stage2_stability_tolerance", 1.2)))


    weights.stab_tolerance = float(stability_tol)


    weights.reward_design = "robust_constrained"
    log(
        f"  {bullet} reward_design={weights.reward_design}"
    )

    user_acc_threshold = float(base_env.acc_threshold)
    if not (np.isfinite(user_acc_threshold) and user_acc_threshold > 0.0):


        new_acc_threshold = _noisy_metric_threshold_from_baseline(
            noisy_baseline_metric=float(noisy_baseline_metric1),
            tolerance=float(allowed_acc_drop),
        )
        base_env.acc_threshold = new_acc_threshold


    if base_env.acc_threshold_m2 is None:
        base_env.acc_threshold_m2 = _noisy_metric_threshold_from_baseline(
            noisy_baseline_metric=float(noisy_baseline_metric2),
            tolerance=float(allowed_acc_drop),
        )


    if base_env.loss_threshold is None:
        base_env.loss_threshold = float(noisy_baseline_loss_mean) * (1.0 + float(allowed_acc_drop))

    stab_floor = float(getattr(weights, "stab_floor", 0.01) or 0.01)
    stab_threshold_m1 = _noisy_std_threshold_from_baseline(
        noisy_baseline_std=float(noisy_baseline_metric1_std),
        stability_multiplier=float(stability_tol),
        floor=stab_floor,
    )
    stab_threshold_m2 = _noisy_std_threshold_from_baseline(
        noisy_baseline_std=float(noisy_baseline_metric2_std),
        stability_multiplier=float(stability_tol),
        floor=stab_floor,
    )
    stab_threshold_loss = _noisy_std_threshold_from_baseline(
        noisy_baseline_std=float(noisy_baseline_loss_std),
        stability_multiplier=float(stability_tol),
        floor=stab_floor,
    )

    user_stab_threshold = float(base_env.stab_threshold)
    stab_calib_summary = ""
    if not np.isfinite(user_stab_threshold):


        base_env.stab_threshold = float(stab_threshold_loss)
        stab_calib_summary = (
            f"multiplier formula: "
            f"loss_std={noisy_baseline_loss_std:.4f} × tol={stability_tol:.4f} "
            f"→ loss_std_threshold={base_env.stab_threshold:.4f}; "
            f"m1_std={noisy_baseline_metric1_std:.4f} × tol={stability_tol:.4f} "
            f"→ m1_std_threshold={stab_threshold_m1:.4f}; "
            f"m2_std={noisy_baseline_metric2_std:.4f} × tol={stability_tol:.4f} "
            f"→ m2_std_threshold={stab_threshold_m2:.4f} "
            f"(floor={stab_floor:.4f})"
        )
    else:
        stab_threshold_loss = float(base_env.stab_threshold)

    baseline_preflight_metrics = {
        "ok": bool(preflight_ok),
        "trial_count": int(getattr(train_cfg, "num_trials_per_step", 1) or 1),
        "metric1_mean": float(noisy_baseline_metric1),
        "metric2_mean": float(noisy_baseline_metric2),
        "loss_mean": float(noisy_baseline_loss_mean),
        "metric1_std": float(noisy_baseline_metric1_std),
        "metric2_std": float(noisy_baseline_metric2_std),
        "loss_std": float(noisy_baseline_loss_std),
        "metric1_threshold": float(base_env.acc_threshold),
        "metric2_threshold": float(base_env.acc_threshold_m2),
        "loss_threshold": (
            float(base_env.loss_threshold) if base_env.loss_threshold is not None else None
        ),
        "metric1_std_threshold": float(stab_threshold_m1),
        "metric2_std_threshold": float(stab_threshold_m2),
        "loss_std_threshold": float(stab_threshold_loss),
        "limit_tolerance": float(allowed_acc_drop),
        "stability_tolerance": float(stability_tol),
        "stability_floor": float(stab_floor),
        "threshold_source": "noisy_all_max_blb_baseline",
    }

    robust_reference = None
    promotion_base_env = None
    authoritative_robust_reference = None
    authoritative_robust_summary = None
    authoritative_validation_banks = None
    authoritative_validation_example_count = 0
    if robust_mode:
        if restored_pending_evidence is None:
            precision_tolerance, stability_multiplier, bootstrap_samples = (
                _resolve_robust_baseline_config(train_cfg, ev)
            )
            if decision_path == "layerwise":
                from rfr.search.rl.stage2.layerwise_runner import (
                    validate_layerwise_validation_bank_config,
                )

                configured_baseline_groups, configured_baseline_trials = (
                    validate_layerwise_validation_bank_config(train_cfg)
                )
            else:
                configured_baseline_groups = int(
                    getattr(train_cfg, "baseline_groups", 5)
                )
                configured_baseline_trials = int(
                    getattr(train_cfg, "baseline_trials_per_group", 3)
                )
            robust_reference, robust_summary = (
                _collect_robust_baseline_reference(
                    base_env=base_env,
                    baseline_action_vec=baseline_action_vec,
                    base_seed=int(train_cfg.seed),
                    precision_tolerance=precision_tolerance,
                    stability_multiplier=stability_multiplier,
                    bootstrap_samples=bootstrap_samples,
                    baseline_groups=configured_baseline_groups,
                    trials_per_group=configured_baseline_trials,
                    max_groups=max(10, 2 * configured_baseline_groups),
                )
            )
        else:
            robust_reference = restored_pending_evidence[
                "robust_reference"
            ]
            robust_summary = dict(
                restored_pending_evidence["baseline_preflight_metrics"].get(
                    "robust_reference", {}
                )
            )
        _install_robust_baseline_reference(
            base_env, baseline, weights, robust_reference,
        )
        base_env.statistical_gate_probability = float(
            getattr(train_cfg, "online_constraint_probability", 0.50)
        )
        stab_floor = float(weights.stab_floor)
        noisy_baseline_loss_mean = float(robust_reference.loss_mean)
        noisy_baseline_metric1 = float(robust_reference.metric1_mean)
        noisy_baseline_metric2 = float(robust_reference.metric2_mean)
        noisy_baseline_loss_std = float(robust_reference.loss_std)
        noisy_baseline_metric1_std = float(robust_reference.metric1_std)
        noisy_baseline_metric2_std = float(robust_reference.metric2_std)
        allowed_acc_drop = float(robust_reference.precision_tolerance)
        stability_tol = float(robust_reference.stability_multiplier)
        stab_threshold_loss = float(robust_reference.loss_std_limit)
        stab_threshold_m1 = float(robust_reference.metric1_std_limit)
        stab_threshold_m2 = float(robust_reference.metric2_std_limit)
        if restored_pending_evidence is None:
            baseline_preflight_metrics["robust_reference"] = robust_summary
            baseline_preflight_metrics.update(robust_summary)
            baseline_preflight_metrics.update({
                "metric1_mean": noisy_baseline_metric1,
                "metric2_mean": noisy_baseline_metric2,
                "loss_mean": noisy_baseline_loss_mean,
                "metric1_std": noisy_baseline_metric1_std,
                "metric2_std": noisy_baseline_metric2_std,
                "loss_std": noisy_baseline_loss_std,
                "metric1_threshold": float(robust_reference.metric1_limit),
                "metric2_threshold": float(robust_reference.metric2_limit),
                "loss_threshold": float(robust_reference.loss_limit),
                "metric1_std_threshold": float(
                    robust_reference.metric1_std_limit
                ),
                "metric2_std_threshold": float(
                    robust_reference.metric2_std_limit
                ),
                "loss_std_threshold": float(
                    robust_reference.loss_std_limit
                ),
                "limit_tolerance": float(
                    robust_reference.precision_tolerance
                ),
                "stability_tolerance": float(
                    robust_reference.stability_multiplier
                ),
                "stability_floor": float(weights.stab_floor),
            })
        else:
            baseline_preflight_metrics = dict(
                restored_pending_evidence["baseline_preflight_metrics"]
            )
        if decision_path == "layerwise":
            (
                promotion_base_env,
                authoritative_validation_example_count,
            ) = _build_search_gate_env(
                runner=runner,
                ev=ev,
                base_env=base_env,
                train_cfg=train_cfg,
                reward_devices=reward_devices,
                log=log,
            )
            if restored_pending_evidence is None:
                bank_references = {}
                bank_summaries = {}
                bank_group_starts = {
                    "A": 1_000, "B": 2_000, "C": 3_000,
                }
                trials_per_bank_group = configured_baseline_trials
                for bank_label in ("A", "B", "C"):
                    bank_reference, bank_summary = (
                        _collect_robust_baseline_reference(
                            base_env=promotion_base_env,
                            baseline_action_vec=baseline_action_vec,
                            base_seed=int(train_cfg.seed),
                            precision_tolerance=precision_tolerance,
                            stability_multiplier=stability_multiplier,
                            bootstrap_samples=bootstrap_samples,
                            baseline_groups=configured_baseline_groups,
                            trials_per_group=trials_per_bank_group,
                            max_groups=configured_baseline_groups,
                            group_index_start=bank_group_starts[bank_label],
                        )
                    )
                    bank_references[bank_label] = bank_reference
                    bank_summaries[bank_label] = bank_summary

                from rfr.search.rl.stage2.layerwise_runner import (
                    LayerwiseValidationBank,
                    LayerwiseValidationBanks,
                )
                from rfr.search.common.statistical_constraints import build_baseline_reference

                promotion_reference = build_baseline_reference(
                    [
                        bank_references["A"].trials,
                        bank_references["B"].trials,
                    ],
                    precision_tolerance=precision_tolerance,
                    stability_multiplier=stability_multiplier,
                    bootstrap_samples=bootstrap_samples,
                    seed=int(train_cfg.seed) + 10_001,
                )
                final_reference = build_baseline_reference(
                    [
                        bank_references["A"].trials,
                        bank_references["B"].trials,
                        bank_references["C"].trials,
                    ],
                    precision_tolerance=precision_tolerance,
                    stability_multiplier=stability_multiplier,
                    bootstrap_samples=bootstrap_samples,
                    seed=int(train_cfg.seed) + 10_002,
                )

                def build_bank(label):
                    summary = bank_summaries[label]
                    return LayerwiseValidationBank(
                        label=label,
                        reference=bank_references[label],
                        probe_seeds=tuple(
                            int(group["group_probe_seed"])
                            for group in summary["groups"]
                        ),
                        trials_per_probe=trials_per_bank_group,
                    )

                authoritative_validation_banks = LayerwiseValidationBanks(
                    bank_a=build_bank("A"),
                    bank_b=build_bank("B"),
                    bank_c=build_bank("C"),
                    promotion_reference=promotion_reference,
                    final_reference=final_reference,
                )
                authoritative_robust_reference = promotion_reference

                def pooled_reference_summary(reference):
                    return {
                        "trial_count": int(reference.trial_count),
                        "loss_mean": float(reference.loss_mean),
                        "metric1_mean": float(reference.metric1_mean),
                        "metric2_mean": float(reference.metric2_mean),
                        "loss_std": float(reference.loss_std),
                        "metric1_std": float(reference.metric1_std),
                        "metric2_std": float(reference.metric2_std),
                        "limits": {
                            "loss": float(reference.loss_limit),
                            "metric1": float(reference.metric1_limit),
                            "metric2": float(reference.metric2_limit),
                            "loss_std": float(reference.loss_std_limit),
                            "metric1_std": float(
                                reference.metric1_std_limit
                            ),
                            "metric2_std": float(
                                reference.metric2_std_limit
                            ),
                        },
                    }

                authoritative_robust_summary = {
                    "ok": True,
                    "schema_version": "stage2_validation_banks_v1",
                    "hard_gate": (
                        "joint_six_point_plus_compute_and_communication_"
                        "counterfactual_six_point_v1"
                    ),
                    "bootstrap_probability_role": "diagnostic_tiebreak_only",
                    "banks": bank_summaries,
                    "promotion_reference_ab": pooled_reference_summary(
                        promotion_reference,
                    ),
                    "final_reference_abc": pooled_reference_summary(
                        final_reference,
                    ),
                    "contract": (
                        authoritative_validation_banks.contract_payload()
                    ),
                }
            else:
                authoritative_validation_banks = (
                    restored_pending_evidence["validation_banks"]
                )
                authoritative_robust_reference = (
                    authoritative_validation_banks.promotion_reference
                )
                authoritative_robust_summary = dict(
                    restored_pending_evidence[
                        "authoritative_robust_summary"
                    ]
                )
                restored_example_count = int(
                    restored_pending_evidence[
                        "authoritative_validation_example_count"
                    ]
                )
                if (
                        int(authoritative_validation_example_count)
                        != restored_example_count
                ):
                    raise RuntimeError(
                        "pending strict runtime validation example count changed"
                    )

            _install_robust_baseline_reference(
                promotion_base_env,
                promotion_base_env.baseline,
                promotion_base_env.reward_weights,
                authoritative_robust_reference,
            )
            authoritative_preflight = {
                **dict(authoritative_robust_summary),
                "split": SEARCH_EVIDENCE_SPLIT,
                "example_count": int(authoritative_validation_example_count),
                "fidelity": "F4",
            }
            if restored_pending_evidence is None:
                baseline_preflight_metrics[
                    "strict_train_probe"
                ] = authoritative_preflight
            elif (
                    baseline_preflight_metrics.get(
                        "strict_train_probe"
                    ) != authoritative_preflight
            ):
                raise RuntimeError(
                    "pending strict authoritative baseline summary changed"
                )

    log(
        f"  {bullet} 基线噪声预热（noisy baseline preflight）："
        f"K={baseline_preflight_metrics['trial_count']}  "
        f"m1(noisy)={noisy_baseline_metric1:.4f}  "
        f"m2(noisy)={noisy_baseline_metric2:.4f}  "
        f"loss_mean(noisy)={noisy_baseline_loss_mean:.4f}  "
        f"std(loss/m1/m2)="
        f"{noisy_baseline_loss_std:.4f}/"
        f"{noisy_baseline_metric1_std:.4f}/"
        f"{noisy_baseline_metric2_std:.4f}"
    )
    _loss_thr_disp = (
        f"{base_env.loss_threshold:.4f}" if base_env.loss_threshold is not None else "None"
    )
    log(
        f"  {bullet} 校准后硬约束阈值（calibrated gates）："
        f"m1_threshold={base_env.acc_threshold:.4f}  "
        f"m2_threshold={float(base_env.acc_threshold_m2):.4f}  "
        f"loss_threshold={_loss_thr_disp}  "
        f"std_thresholds(loss/m1/m2)="
        f"{stab_threshold_loss:.4f}/"
        f"{stab_threshold_m1:.4f}/"
        f"{stab_threshold_m2:.4f}  "
        f"(limit_tol={allowed_acc_drop:.4f}, stab_tol={stability_tol:.4f}; "
        f"loss 越低越好，允许相对上浮 limit_tol；"
        f"m1/m2 越高越好，允许相对下降 limit_tol；std 越低越好，× stab_tol)"
    )
    if stab_calib_summary:
        log(f"  {bullet} 稳定阈值校准来源（stab calibration source）：{stab_calib_summary}")

    if reward_devices and len(reward_devices) >= 2:
        try:
            base_env.clear_installed_blb()
        except Exception:
            pass
        base_env.env_cfg.persistent_probe_install = True
        log(
            f"  {bullet} Multi-GPU BLB install cache：enabled "
            f"(devices={reward_devices}; wrappers/hooks stay installed and cfgs update in-place)"
        )

    _seq_block_title(log, "基线信号（baseline cost / reward / metrics）")
    _seq_log_rounded_box(log, [
        f"成本基线（baseline cost）："
        f"total_bits={baseline.total_bits_sum}, "
        f"fusion={baseline.total_fusion_count}, "
        f"avg_k={baseline.avg_k:.2f}",
        f"指标基线（baseline metrics）："
        f"loss={baseline.loss_mean:.4f}, m1={baseline.metric1_mean:.4f}, m2={baseline.metric2_mean:.4f}",
        f"奖励权重（reward weights, v2-style rdv2）："
        f"cost_weight={weights.cost_weight:.4g}, lambda_stab={weights.lambda_stab:.4g}, "
        f"invalid_penalty={weights.invalid_penalty:.4g}, "
        f"clip=[{weights.reward_clip_min:.1f}, {weights.reward_clip_max:.1f}], "
        f"tier=[{weights.tier_metric_bonus:.1f}, +{weights.tier_stability_bonus:.1f}], "
        f"baseline_metric1={weights.baseline_metric1:.4f}",
        f"硬约束阈值（acc=hard, stab=soft cap for excess penalty）："
        f"m1_threshold={base_env.acc_threshold:.4f}, "
        f"m2_threshold={float(base_env.acc_threshold_m2):.4f}, "
        f"loss_threshold={_loss_thr_disp}, "
        f"std_thresholds(loss/m1/m2)="
        f"{stab_threshold_loss:.4f}/{stab_threshold_m1:.4f}/{stab_threshold_m2:.4f}",
        f"static_skeletons archive：{ss_baseline_obj.archive_path}",
    ])


    base_env.pareto_cost_archive = ParetoCostArchive(baseline=baseline)
    log(
        f"  {bullet} Adaptive scalar cost reward：P1/P2 不吃 cost；P3 中 "
        f"fusion/K 使用区间式 boost，total_bits 使用弱线性项；Pareto frontier 仅用于诊断/探索统计。"
    )

    from rfr.preparation.fusion.count_map import FusionCountMap

    fusion_map = FusionCountMap.load(str(train_cfg.profile))
    log(
        f"  {bullet} Fusion-count action: graphs={len(fusion_map.graphs)}, "
        f"max_options={fusion_map.max_num_options()}"
    )
    if int(train_cfg.num_trials_per_step) < 2:
        raise ValueError("layerwise stability requires at least two probe trials")
    return _run_layerwise_training_branch(
            train_cfg=train_cfg,
            evaluator=ev,
            base_env=base_env,
            fusion_map=fusion_map,
            max_sfs=max_sfs,
            robust_reference=robust_reference,
            promotion_base_env=promotion_base_env,
            authoritative_robust_reference=authoritative_robust_reference,
            authoritative_robust_summary=authoritative_robust_summary,
            authoritative_validation_banks=authoritative_validation_banks,
            authoritative_validation_example_count=(
                authoritative_validation_example_count
            ),
            static_skeletons_baseline=ss_baseline_obj,
            baseline_action_vec=ss_action_vec,
            fixed_gelu=fixed_gelu,
            fixed_softmax=fixed_softmax,
            fixed_label=fixed_label,
            fixed_source=fixed_source,
            blb_progress_dir=blb_progress_dir,
            clean_baseline_metrics=baseline_metrics,
            baseline_preflight_metrics=baseline_preflight_metrics,
            status=status,
            resume_checkpoint_path=resume_checkpoint_path,
            run_lock=run_lock,
            log=log,
    )
