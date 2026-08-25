"""Production entrypoint for layerwise Stage-2 training and comparators."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import sys
from typing import Any, Mapping, Optional

import numpy as np
import torch

from rfr.preparation.data.protocol import TRAIN_PROBE_SIZE, TRAIN_PROBE_SPLIT
from rfr.preparation.rescale.bridge import (
    RescaleOptimizerBridge,
    build_rescale_invoker,
)

from .env import ProbeBatch
from .sequential_policy import SequentialPPOConfig


BLB_STAGE2_LIVE_CHECKPOINT_FILENAME = "blb_stage2_rl_checkpoint_live.pt"
BLB_STAGE2_FINAL_CHECKPOINT_FILENAME = "blb_stage2_rl_checkpoint_final.pt"
BLB_STAGE2_BEST_CFG_FILENAME = "blb_stage2_best_cfg.pkl"


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_blb_persistence_dir(evaluator: Any) -> str:
    run_dir = str(getattr(evaluator, "run_output_dir", "") or "").strip()
    if run_dir:
        if getattr(evaluator, "decoupled_layout", False):
            output_dir = os.path.join(run_dir, "progress")
        else:
            output_dir = os.path.join(run_dir, "stage2_noise", "progress")
    else:
        output_dir = os.path.join(
            _repo_root(),
            "Parting Chapter",
            "persistent",
            "blb_stage2_default_run",
            "stage2_noise",
            "progress",
        )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _effective_stage2_inference_batch_size(
    evaluator: Any,
    config: "BLBStage2TrainConfig",
) -> int:
    value = config.stage2_inference_batch_size
    if value in (None, ""):
        value = getattr(evaluator, "stage2_inference_batch_size", None)
    if value in (None, ""):
        value = getattr(evaluator, "batch_size", 1)
    value = int(value)
    if value <= 0:
        raise ValueError("stage2_inference_batch_size must be positive")
    return value


def _selection_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _build_best_noise_config(evaluator: Any) -> dict[str, np.ndarray]:
    """Keep the established all-max fields consumed by final evaluation."""
    return evaluator._get_max_noise_configuration()


@dataclass
class BLBStage2TrainConfig:
    total_episodes: int = 0
    rollout_size: int = 120
    seed: int = 42
    eval_interval: int = 100
    save_interval: int = 200
    profile: str = "default"
    acc_threshold: float = 0.0
    stab_threshold: float = float("inf")
    ppo: SequentialPPOConfig = field(default_factory=SequentialPPOConfig)

    search_backend: str = "ppo"
    search_evaluation_budget: int = 0
    search_initial_design_size: int = 64
    search_candidate_pool_size: int = 2048
    search_population_size: int = 64
    search_patience_generations: int = 100
    search_mutation_max_coordinates: int = 3
    search_rf_n_estimators: int = 128
    search_rf_min_samples_leaf: int = 2
    search_full_validation: bool = True

    num_trials_per_step: int = 3
    probe_batch_count: int = 4
    stage2_inference_batch_size: Optional[int] = None
    truncation_backend: str = "binary"
    truncation_ring_bits: int = 43
    truncation_source_fractional_bits: int = 24
    calibrate_baseline_samples: int = 8
    inproc_rescale_optimizer_root: str = field(
        default_factory=lambda: os.path.join(_repo_root(), "configs/preparation/rescale")
    )
    inproc_baseline_archive: Optional[str] = None

    reward_devices: list[int] = field(default_factory=list)
    online_num_trials_per_step: int = 3
    terminal_eval_batch_size: int = 4
    promotion_validation_trials: int = 15
    promotion_margin_window: float = 0.25
    final_selection_top_n: int = 20
    final_selection_validation_trials: int = 15
    baseline_groups: int = 5
    baseline_trials_per_group: int = 3
    constraint_bootstrap_samples: int = 4096
    online_constraint_probability: float = 0.50
    promotion_constraint_probability: float = 0.80
    final_constraint_probability: float = 0.95
    stage2_stability_multiplier: float = 2.0
    communication_importance_ratio: float = 1.0
    convergence_min_episodes: int = 90_000
    convergence_patience_updates: int = 100

    def __post_init__(self) -> None:
        from rfr.search.common.precision_presets import validate_communication_importance_ratio
        from .search_baselines import normalize_search_backend

        self.search_backend = normalize_search_backend(self.search_backend)
        self.total_episodes = int(self.total_episodes)
        if self.total_episodes < 0:
            raise ValueError("total_episodes must be nonnegative")
        self.rollout_size = int(self.rollout_size)
        if self.rollout_size <= 0:
            raise ValueError("rollout_size must be positive")
        self.seed = int(self.seed)
        self.search_evaluation_budget = int(self.search_evaluation_budget)
        if self.search_evaluation_budget < 0:
            raise ValueError("search_evaluation_budget must be nonnegative")
        for name in (
            "search_initial_design_size",
            "search_candidate_pool_size",
            "search_population_size",
            "search_patience_generations",
            "search_mutation_max_coordinates",
            "search_rf_n_estimators",
            "search_rf_min_samples_leaf",
            "num_trials_per_step",
            "calibrate_baseline_samples",
            "online_num_trials_per_step",
            "terminal_eval_batch_size",
            "promotion_validation_trials",
            "final_selection_top_n",
            "final_selection_validation_trials",
            "baseline_groups",
            "baseline_trials_per_group",
            "constraint_bootstrap_samples",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            setattr(self, name, value)
        if self.stage2_inference_batch_size not in (None, ""):
            self.stage2_inference_batch_size = int(self.stage2_inference_batch_size)
            if self.stage2_inference_batch_size <= 0:
                raise ValueError("stage2_inference_batch_size must be positive")
        if self.truncation_backend != "binary":
            raise ValueError("production Stage-2 requires binary truncation")
        if int(self.truncation_ring_bits) != 43:
            raise ValueError("production Stage-2 requires ring width 43")
        if int(self.truncation_source_fractional_bits) != 24:
            raise ValueError("production Stage-2 requires 24 source fractional bits")
        self.stage2_stability_multiplier = float(self.stage2_stability_multiplier)
        if self.stage2_stability_multiplier <= 0.0:
            raise ValueError("stage2_stability_multiplier must be positive")
        self.communication_importance_ratio = validate_communication_importance_ratio(
            self.communication_importance_ratio
        )
        probabilities = (
            float(self.online_constraint_probability),
            float(self.promotion_constraint_probability),
            float(self.final_constraint_probability),
        )
        if not (0.0 < probabilities[0] <= probabilities[1] <= probabilities[2] <= 1.0):
            raise ValueError(
                "constraint probabilities must satisfy 0 < online <= promotion <= final <= 1"
            )
        (
            self.online_constraint_probability,
            self.promotion_constraint_probability,
            self.final_constraint_probability,
        ) = probabilities
        self.convergence_min_episodes = int(self.convergence_min_episodes)
        self.convergence_patience_updates = int(self.convergence_patience_updates)
        if self.convergence_min_episodes < 90_000:
            raise ValueError("convergence_min_episodes must be at least 90000")
        if self.convergence_patience_updates < 100:
            raise ValueError("convergence_patience_updates must be at least 100")


class BLBStage2RLRunner:
    """Build the fixed production config and execute layerwise Stage 2."""

    def __init__(self, evaluator: Any):
        self.evaluator = evaluator

    def run(
        self,
        fixed_gelu: Any,
        fixed_softmax: Any,
        fixed_label: str,
        fixed_source: str,
        resume_checkpoint_path: Optional[str] = None,
    ) -> dict[str, Any]:
        config = self._build_train_config_from_evaluator(self.evaluator)
        return self._run_with_config(
            config,
            fixed_gelu=fixed_gelu,
            fixed_softmax=fixed_softmax,
            fixed_label=fixed_label,
            fixed_source=fixed_source,
            resume_checkpoint_path=resume_checkpoint_path,
        )

    def _run_with_config(
        self,
        config: BLBStage2TrainConfig,
        *,
        fixed_gelu: Any,
        fixed_softmax: Any,
        fixed_label: str,
        fixed_source: str,
        resume_checkpoint_path: Optional[str] = None,
    ) -> dict[str, Any]:
        from .sequential_runner import run_sequential_via_runner

        self.evaluator.activate_stage2_inference_batch_size()
        return run_sequential_via_runner(
            runner=self,
            train_cfg=config,
            fixed_gelu=fixed_gelu,
            fixed_softmax=fixed_softmax,
            fixed_label=fixed_label,
            fixed_source=fixed_source,
            resume_checkpoint_path=resume_checkpoint_path,
        )

    def _build_train_config_from_evaluator(self, evaluator: Any) -> BLBStage2TrainConfig:
        from rfr.preparation.rescale.baseline_bootstrap import resolve_stage2_profile
        from rfr.search.runtime.probe_runner import parse_device_ids

        config = BLBStage2TrainConfig(
            total_episodes=int(getattr(evaluator, "stage2_rl_episodes", 0)),
            seed=int(getattr(evaluator, "blb_v3_seed", None) or 42),
            profile=resolve_stage2_profile(
                str(evaluator.dataset_key),
                model_type=str(getattr(evaluator, "model_type", "")),
                num_layers=int(evaluator.total_layers),
            ),
            num_trials_per_step=int(getattr(evaluator, "stage2_k_trials", 3)),
            stage2_inference_batch_size=getattr(
                evaluator, "stage2_inference_batch_size", None
            ),
            reward_devices=parse_device_ids(
                getattr(evaluator, "blb_v3_reward_devices", None)
            ),
            inproc_rescale_optimizer_root=str(
                getattr(
                    evaluator,
                    "blb_v3_inproc_rescale_optimizer_root",
                    os.path.join(_repo_root(), "configs/preparation/rescale"),
                )
                or os.path.join(_repo_root(), "configs/preparation/rescale")
            ),
            search_backend=getattr(evaluator, "blb_v3_search_backend", "ppo"),
            search_evaluation_budget=int(
                getattr(evaluator, "blb_v3_search_evaluation_budget", 0)
            ),
            search_initial_design_size=int(
                getattr(evaluator, "blb_v3_search_initial_design_size", 64)
            ),
            search_candidate_pool_size=int(
                getattr(evaluator, "blb_v3_search_candidate_pool_size", 2048)
            ),
            search_population_size=int(
                getattr(evaluator, "blb_v3_search_population_size", 64)
            ),
            search_patience_generations=int(
                getattr(evaluator, "blb_v3_search_patience_generations", 100)
            ),
            search_mutation_max_coordinates=int(
                getattr(evaluator, "blb_v3_search_mutation_max_coordinates", 3)
            ),
            search_rf_n_estimators=int(
                getattr(evaluator, "blb_v3_search_rf_n_estimators", 128)
            ),
            search_rf_min_samples_leaf=int(
                getattr(evaluator, "blb_v3_search_rf_min_samples_leaf", 2)
            ),
            search_full_validation=bool(
                getattr(evaluator, "blb_v3_search_full_validation", True)
            ),
            rollout_size=int(getattr(evaluator, "blb_v3_rollout_size", None) or 120),
            eval_interval=int(getattr(evaluator, "blb_v3_eval_interval", None) or 100),
            save_interval=int(getattr(evaluator, "blb_v3_save_interval", None) or 200),
            calibrate_baseline_samples=int(
                getattr(evaluator, "blb_v3_calibrate_baseline_samples", None) or 8
            ),
            online_num_trials_per_step=int(
                getattr(evaluator, "blb_v3_online_k_trials", 3)
            ),
            terminal_eval_batch_size=int(
                getattr(evaluator, "blb_v3_terminal_eval_batch_size", 4)
            ),
            promotion_validation_trials=int(
                getattr(evaluator, "blb_v3_promotion_validation_trials", 15)
            ),
            promotion_margin_window=float(
                getattr(evaluator, "blb_v3_promotion_margin_window", 0.25)
            ),
            final_selection_top_n=int(
                getattr(evaluator, "blb_v3_final_selection_top_n", 20)
            ),
            final_selection_validation_trials=int(
                getattr(evaluator, "blb_v3_final_selection_validation_trials", 15)
            ),
            baseline_groups=int(getattr(evaluator, "blb_v3_baseline_groups", 5)),
            baseline_trials_per_group=int(
                getattr(evaluator, "blb_v3_baseline_trials_per_group", 3)
            ),
            constraint_bootstrap_samples=int(
                getattr(evaluator, "blb_v3_constraint_bootstrap_samples", 4096)
            ),
            online_constraint_probability=float(
                getattr(evaluator, "blb_v3_online_constraint_probability", 0.50)
            ),
            promotion_constraint_probability=float(
                getattr(evaluator, "blb_v3_promotion_constraint_probability", 0.80)
            ),
            final_constraint_probability=float(
                getattr(evaluator, "blb_v3_final_constraint_probability", 0.95)
            ),
            stage2_stability_multiplier=float(
                getattr(evaluator, "stage2_stability_multiplier", 2.0)
            ),
            communication_importance_ratio=float(
                getattr(evaluator, "stage2_communication_importance_ratio", 1.0)
            ),
            convergence_min_episodes=int(
                getattr(evaluator, "blb_v3_min_convergence_episodes", 90_000)
            ),
            convergence_patience_updates=int(
                getattr(evaluator, "blb_v3_convergence_patience_updates", 100)
            ),
        )
        config.ppo.lr = float(
            getattr(evaluator, "stage2_ppo_lr_initial", config.ppo.lr)
        )
        config.stage2_inference_batch_size = _effective_stage2_inference_batch_size(
            evaluator, config
        )
        if config.total_episodes > 0:
            config.rollout_size = min(config.rollout_size, config.total_episodes)
        if config.search_backend != "ppo" and config.search_evaluation_budget <= 0:
            raise ValueError("comparator search requires a positive evaluation budget")
        return config

    def _build_probe_batches(
        self,
        evaluator: Any,
        config: BLBStage2TrainConfig,
        *,
        probe_size_override: Optional[int] = None,
    ) -> list[ProbeBatch]:
        requested = int(
            getattr(evaluator, "stage2_probe_size", TRAIN_PROBE_SIZE)
            if probe_size_override is None
            else probe_size_override
        )
        if requested != TRAIN_PROBE_SIZE:
            raise ValueError(
                f"Stage-2 requires the fixed {TRAIN_PROBE_SIZE}-example train probe"
            )
        splits = getattr(evaluator, "dataset_splits", None)
        if not isinstance(splits, Mapping):
            raise RuntimeError("Stage-2 requires evaluator.dataset_splits")
        probe = splits.get(TRAIN_PROBE_SPLIT)
        if probe is None or len(probe) != TRAIN_PROBE_SIZE:
            raise RuntimeError(
                f"Stage-2 {TRAIN_PROBE_SPLIT} must contain {TRAIN_PROBE_SIZE} examples"
            )
        from torch.utils.data import DataLoader

        loader = DataLoader(
            probe,
            batch_size=_effective_stage2_inference_batch_size(evaluator, config),
            shuffle=False,
            drop_last=False,
            collate_fn=evaluator.data_collator,
            pin_memory=torch.cuda.is_available(),
        )
        device = torch.device(evaluator.device)
        return [ProbeBatch.from_batch(batch, device) for batch in loader]

    def _build_rescale_bridge(
        self,
        config: BLBStage2TrainConfig,
        log: Any,
    ) -> RescaleOptimizerBridge:
        root = str(config.inproc_rescale_optimizer_root)
        profile = str(config.profile)
        log(f"  * Rescale_optimizer root: {root}")
        try:
            invoker = build_rescale_invoker(
                root=root,
                profile=profile,
                baseline_archive=config.inproc_baseline_archive,
            )
        except Exception as exc:
            raise RuntimeError(
                f"failed to initialize in-process Rescale for {profile}: {exc}"
            ) from exc
        return RescaleOptimizerBridge(invoker=invoker)

    @staticmethod
    def _estimate_baseline_metrics(environment: Any) -> Any:
        return environment._eval_on_probe(environment.env_cfg.num_trials_per_step)

    @staticmethod
    def _make_log_safe(log_fn: Any) -> Any:
        def safe_log(message: Any) -> Any:
            try:
                return log_fn(message)
            except UnicodeEncodeError:
                encoding = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
                safe = str(message).encode(encoding, errors="replace").decode(encoding)
                return log_fn(safe)

        return safe_log


def run_stage2_search(
    evaluator: Any,
    fixed_stage1: Mapping[str, Any],
    config: BLBStage2TrainConfig,
) -> dict[str, Any]:
    """Execute the production Stage-2 path from an explicit fixed Stage-1 result."""
    return BLBStage2RLRunner(evaluator)._run_with_config(
        config,
        fixed_gelu=fixed_stage1["gelu"],
        fixed_softmax=fixed_stage1["softmax"],
        fixed_label=str(fixed_stage1.get("label", "Stage-1 result")),
        fixed_source=str(fixed_stage1.get("source", "stage1_result")),
    )
