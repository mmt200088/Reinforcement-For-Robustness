"""Cross-stage workflow, model-evaluation context and JSON handoffs.

Stage-1 algorithms live in search.rl.stage1; this callback owns shared model,
data and persistent state and dispatches the selected production stages."""

from __future__ import annotations

from collections import deque
import time
import math
import sys
import threading
from typing import Any, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers.trainer_callback import TrainerCallback
from sklearn.metrics import accuracy_score, f1_score
from rfr.search.common.eval_metrics import (
    summarize_eval_trials,
)
from rfr.search.runtime.inference_eval import (
    normalize_labels_for_metrics as shared_normalize_labels_for_metrics,
    normalize_logits_for_metrics as shared_normalize_logits_for_metrics,
    run_installed_model_on_dataloader,
)
from rfr.search.runtime.model_handler import (
    ReversibleLayerHandler,
)
from rfr.preparation.data.protocol import TRAIN_PROBE_SPLIT, validate_dataset
from rfr.preparation.data.mrpc_contract import (
    MRPC_STAGE2_RL_ALIGNMENT_BATCH_SIZE,
)
from rfr.search.common.data_points import (
    write_dataset_protocol,
)
import os
import hashlib
from rfr.search.rl.stage1 import config as stage1_config
from rfr.search.rl.stage1.config import resolve_ppo_learning_rate
from rfr.search.rl.stage1.policy import RunningMeanStd
from rfr.search.common.run_artifacts import (
    SEARCH_LOG_HEADER,
    ensure_parent_dir,
    resolve_run_output_layout,
    update_persistent_metadata_stage,
)
from rfr.search.rl.stage1.runner import run_stage1

def _stage1_changed_layer_indices(current_degrees, previous_degrees):
    if previous_degrees is None or len(previous_degrees) != len(current_degrees):
        return range(len(current_degrees))
    return [
        idx for idx, (current, previous) in enumerate(
            zip(current_degrees, previous_degrees)
        )
        if current != previous
    ]


def _install_stage1_function_configuration(
        handler, handler_layer_name, cfg_sig, previous_cfg=None):
    gelu_degrees, softmax_degrees = cfg_sig
    previous_gelu = previous_softmax = None
    if isinstance(previous_cfg, (list, tuple)) and len(previous_cfg) == 2:
        previous_gelu, previous_softmax = previous_cfg

    gelu_indices = _stage1_changed_layer_indices(gelu_degrees, previous_gelu)
    softmax_indices = _stage1_changed_layer_indices(
        softmax_degrees, previous_softmax,
    )

    original_gelu_layers = [
        idx for idx in gelu_indices
        if gelu_degrees[idx] == stage1_config.STAGE1_ORIGINAL_FUNCTION_DEGREE
    ]
    if original_gelu_layers:
        handler.restore_layer_gelu(original_gelu_layers, handler_layer_name)

    original_softmax_layers = [
        idx for idx in softmax_indices
        if softmax_degrees[idx] == stage1_config.STAGE1_ORIGINAL_FUNCTION_DEGREE
    ]
    if original_softmax_layers:
        handler.restore_layer_softmax(original_softmax_layers, handler_layer_name)

    gelu_map = {degree: [] for degree in (0, 1, 2, 4)}
    for idx in gelu_indices:
        degree = gelu_degrees[idx]
        if degree in gelu_map:
            gelu_map[degree].append(idx)
    for degree in (0, 1, 2, 4):
        if gelu_map[degree]:
            handler.replace_layer_gelu(
                gelu_map[degree], handler_layer_name, degree=degree,
            )

    softmax_map = {degree: [] for degree in range(2, 7)}
    for idx in softmax_indices:
        degree = softmax_degrees[idx]
        if degree in softmax_map:
            softmax_map[degree].append(idx)
    for degree in range(2, 7):
        if softmax_map[degree]:
            handler.replace_layer_softmax(
                softmax_map[degree], handler_layer_name, degree=degree,
            )


DATASET_METRICS_CONFIG = {
    dataset: {
        'type': 'classification',
        'metrics': ['accuracy', 'weighted_f1'],
        'metric_names': ['Acc.', 'F1'],
        'metric_full_names': ['Accuracy', 'Weighted F1'],
    }
    for dataset in ('mrpc', 'rte', 'sst2')
}


class SearchWorkflow(TrainerCallback):
    def __init__(self, model, train_data, test_data, data_collator, rl_lr=None, degree=None,
                 batch_size=16,
                 stage1_rl_episodes=stage1_config.PPO_MAX_EPISODES,
                 stage2_rl_episodes=0,
                 stage1_rl_episodes_specified=False,
                 stage2_rl_episodes_specified=False,
                 stage1_rl_lr=None,
                 stage2_rl_lr=None,
                 stage1_rl_devices="",
                 device='cuda', data_path='mrpc',
                 run_output_dir='',
                 stage1_best_config_path='',
                 search_best_config_path='',
                 random_seed=42,
                 final_eval_repeat_n=5,
                 skip_noise_rl=False,
                 skip_stage1_rl=False,
                 skip_final_eval=True,
                 final_eval_only=False,
                 resume_run_dir='',
                 stage1_accuracy_tolerance=None,
                 stage2_limit_tolerance=None,
                 stage2_stability_tolerance=None,
                 stage2_stability_multiplier=2.0,
                 stage2_communication_importance_ratio=1.0,
                 stage2_k_trials=None,
                 stage2_probe_size=None,
                 stage2_inference_batch_size=None,
                 blb_v3_rollout_size=None,
                  blb_v3_eval_interval=None,
                  blb_v3_save_interval=None,
                  blb_v3_calibrate_baseline_samples=None,
                  blb_v3_inproc_rescale_optimizer_root=None,
                  blb_v3_seed=None,
                  blb_v3_reward_devices="",
                  blb_v3_online_k_trials=3,
                  blb_v3_terminal_eval_batch_size=4,
                  blb_v3_promotion_validation_trials=15,
                  blb_v3_final_selection_top_n=20,
                  blb_v3_final_selection_validation_trials=15,
                  blb_v3_promotion_margin_window=0.25,
                  blb_v3_baseline_groups=5,
                  blb_v3_baseline_trials_per_group=3,
                  blb_v3_constraint_bootstrap_samples=4096,
                  blb_v3_online_constraint_probability=0.50,
                  blb_v3_promotion_constraint_probability=0.80,
                  blb_v3_final_constraint_probability=0.95,
                  blb_v3_search_backend="ppo",
                  blb_v3_search_initial_design_size=64,
                  blb_v3_search_candidate_pool_size=2048,
                  blb_v3_search_population_size=64,
                  blb_v3_search_mutation_max_coordinates=3,
                  blb_v3_search_rf_n_estimators=128,
                  blb_v3_search_rf_min_samples_leaf=2,
                  comparator_bo_stage1_no_improvement=1000,
                  comparator_bo_stage2_no_improvement=2000,
                  comparator_greedy_stage1_no_improvement_rounds=1,
                  comparator_greedy_stage2_no_improvement_rounds=1,
                  comparator_ga_stage1_generations=200,
                  comparator_ga_stage2_generations=200,
                  comparator_stage1_only=False,
                  glue_data_protocol=None):
        """Construct the shared Stage-1, Stage-2, and final-eval pipeline."""

        sys.setrecursionlimit(50000)

        self.model = model
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.data_collator = data_collator
        self.batch_size = max(1, int(batch_size))
        self._active_inference_batch_size = self.batch_size
        self.stage2_inference_batch_size = (
            None
            if stage2_inference_batch_size in (None, "")
            else self._coerce_positive_int(
                stage2_inference_batch_size, "stage2_inference_batch_size",
            )
        )
        self.glue_data_protocol = glue_data_protocol
        self.dataset_protocol = (
            self.glue_data_protocol.as_payload()
            if self.glue_data_protocol is not None
            else None
        )
        self.dataset_protocol_hash = (
            self.glue_data_protocol.dataset_protocol_hash
            if self.glue_data_protocol is not None
            else None
        )
        if (
                self.glue_data_protocol is not None
                and self.glue_data_protocol.validation_full is not test_data
        ):
            raise ValueError(
                "validation data does not match the GLUE protocol context"
            )
        from rfr.search.rl.stage1.eval_cache import Stage1EvalCache
        self._eval_cache = Stage1EvalCache()


        self._stage1_worker_eval_cache = Stage1EvalCache()
        self._stage1_parallel_timing_lock = threading.Lock()
        self._stage1_parallel_model_forward_seconds = 0.0
        self._stage1_parallel_model_forward_calls = 0

        self._eval_infra_ready = False


        self._last_applied_config = None
        self.stage1_rl_episodes = self._coerce_positive_int(
            stage1_rl_episodes, 'stage1_rl_episodes'
        )
        self.stage1_rl_episode_limit = int(self.stage1_rl_episodes)
        self.stage2_rl_episodes = self._coerce_nonnegative_int(
            stage2_rl_episodes, 'stage2_rl_episodes'
        )
        self.stage1_rl_episodes_specified = self._coerce_bool_flag(
            stage1_rl_episodes_specified, 'stage1_rl_episodes_specified'
        )
        self.stage2_rl_episodes_specified = self._coerce_bool_flag(
            stage2_rl_episodes_specified, 'stage2_rl_episodes_specified'
        )


        self.data_path = data_path
        self.is_regression = self._detect_task_type()
        self._log_task_type()


        _pin = torch.cuda.is_available()
        self.dataloader_train = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=data_collator,
            pin_memory=_pin,
        )
        self.dataloader_test = DataLoader(
            test_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=data_collator,
            pin_memory=_pin,
        )


        self._prepare_rl_datasets(
            train_data=train_data,
            train_probe=(
                self.glue_data_protocol.train_probe
                if self.glue_data_protocol is not None
                else train_data
            ),
            validation_data=test_data,
        )

        try:
            self.reversible_handler = ReversibleLayerHandler(self.model)
        except Exception as e:
            print(f"[警告] 深拷贝（Deepcopy）在处理器初始化时失败: {e}。'restore_all' 可能会失败。")
            self.reversible_handler = ReversibleLayerHandler(self.model)

        self.layers_attribute = self._detect_layer_attribute()
        self.total_layers = len(eval('self.model.' + self.layers_attribute))


        self.GELU_COST_MAP = {4: 3.0, 2: 2.5, 1: 1.0, 0: -1.0}
        self.SOFTMAX_COST_MAP = {6: 3.0, 5: 2.5, 4: 2.0, 3: 1.5, 2: 1.0}


        self.current_gelu_degrees = np.full(self.total_layers, 4, dtype=int)
        self.current_softmax_degrees = np.full(self.total_layers, 6, dtype=int)
        self.rl_lr_raw = rl_lr
        self.stage1_rl_lr_raw = stage1_rl_lr if stage1_rl_lr not in (None, "") else rl_lr
        self.stage2_rl_lr_raw = stage2_rl_lr if stage2_rl_lr not in (None, "") else self.stage1_rl_lr_raw
        self.stage1_ppo_lr_initial, self.stage1_ppo_lr_mode = resolve_ppo_learning_rate(
            self.stage1_rl_lr_raw
        )
        self.stage2_ppo_lr_initial, self.stage2_ppo_lr_mode = resolve_ppo_learning_rate(
            self.stage2_rl_lr_raw,
            default_lr=self.stage1_ppo_lr_initial,
        )
        self.ppo_lr_initial = self.stage1_ppo_lr_initial
        self.ppo_lr_mode = self.stage1_ppo_lr_mode


        _s1_tol = float(stage1_accuracy_tolerance) if stage1_accuracy_tolerance is not None else 0.005
        self.error_threshold = _s1_tol
        self.correlation_drop_ratio = _s1_tol


        self.stage2_limit_tolerance = float(stage2_limit_tolerance) if stage2_limit_tolerance is not None else 0.05


        self.stage2_stability_tolerance = float(stage2_stability_tolerance) if stage2_stability_tolerance is not None else 1.2
        self.stage2_stability_multiplier = float(stage2_stability_multiplier)
        if self.stage2_stability_multiplier <= 0.0:
            raise ValueError("stage2_stability_multiplier must be positive")
        from rfr.search.common.precision_presets import (
            validate_communication_importance_ratio,
        )

        self.stage2_communication_importance_ratio = (
            validate_communication_importance_ratio(
                stage2_communication_importance_ratio,
            )
        )


        self.stage2_k_trials = max(1, int(stage2_k_trials)) if stage2_k_trials is not None else 3
        self.stage2_probe_size = max(1, int(stage2_probe_size)) if stage2_probe_size is not None else 256

        self.search_algorithm = "rl"

        self.standalone_stage_layout = (
            str(blb_v3_search_backend or "ppo").strip().lower() == "ppo"
        )
        output_layout = resolve_run_output_layout(
            run_output_dir,
            flattened=self.standalone_stage_layout,
        )
        self.run_output_dir = output_layout["run_output_dir"]
        self.dataset_protocol_path = None
        if self.dataset_protocol is not None and self.run_output_dir:
            self.dataset_protocol_path = write_dataset_protocol(
                self.run_output_dir,
                self.dataset_protocol,
            )
        self.log_file = output_layout["log_file"]
        self.noise_log_file = output_layout["noise_log_file"]
        self.active_log_file = self.log_file
        self.step_info_file = output_layout["stage1_step_info_file"]
        self.stage1_step_info_file = self.step_info_file
        self.stage1_training_curve_path = output_layout["stage1_training_curve_path"]
        self.stage1_entropy_curve_path = output_layout["stage1_entropy_curve_path"]
        self.final_eval_dir = output_layout["final_eval_dir"]
        self.noise_step_info_file = output_layout["noise_step_info_file"]
        self.noise_stage_training_curve_path = output_layout["noise_training_curve_path"]
        self.noise_stage_entropy_curve_path = output_layout["noise_entropy_curve_path"]
        self.noise_stage_progress_dir = output_layout["noise_progress_dir"]
        self._noise_log_initialized = self.noise_log_file == self.log_file
        _log_header = SEARCH_LOG_HEADER
        ensure_parent_dir(self.log_file)

        _is_resuming = bool(str(resume_run_dir or '').strip())
        _log_open_mode = "a" if _is_resuming and os.path.isfile(self.log_file) else "w"
        with open(self.log_file, _log_open_mode, encoding="utf-8") as f:
            if _is_resuming:
                f.write(f"\n{'='*60}\n")
                import datetime as _dt
                f.write(f"=== 续训日志开始（Resume Log Started） {_dt.datetime.now().isoformat()} ===\n")
                f.write(f"{'='*60}\n")
            else:
                f.write(_log_header + "\n")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(
                f"[信息] Stage-1 PPO学习率（LR）从 stage1_rl_lr={self.stage1_rl_lr_raw!r} 解析为 -> "
                f"{self.stage1_ppo_lr_initial:.6g} ({self.stage1_ppo_lr_mode}) | "
                f"Stage-2 PPO学习率（LR）从 stage2_rl_lr={self.stage2_rl_lr_raw!r} 解析为 -> "
                f"{self.stage2_ppo_lr_initial:.6g} ({self.stage2_ppo_lr_mode})\n"
            )
            f.write(
                f"[信息] 第一阶段RL回合数（Stage-1 RL episodes）: {self.stage1_rl_episodes} | "
                f"第二阶段RL回合数（Stage-2 RL episodes）: {self.stage2_rl_episodes}\n"
            )
            if self.run_output_dir:
                f.write(f"[信息] 统一运行输出目录（Unified run output dir）: {self.run_output_dir}\n")


        self.current_episode = 0
        self.total_episodes = self.stage1_rl_episode_limit
        self.current_entropy_coef = stage1_config.PPO_ENTROPY_INITIAL
        self.current_lr = self.stage1_ppo_lr_initial


        self.reward_history = deque(maxlen=stage1_config.RUNNING_REWARD_HISTORY_SIZE)
        self.reward_history_sum = 0.0
        self.reward_history_sumsq = 0.0
        self.reward_mean = 0.0
        self.reward_std = 1.0


        self.return_normalizer = RunningMeanStd()


        self.stage1_best_config_input_path = str(
            stage1_best_config_path or ""
        ).strip()
        self.search_best_config_input_path = str(
            search_best_config_path or ""
        ).strip()
        self.random_seed = int(random_seed)
        self.final_eval_repeat_n = self._coerce_positive_int(
            final_eval_repeat_n, "final_eval_repeat_n",
        )
        self.skip_stage1_rl = self._coerce_bool_flag(
            skip_stage1_rl, "skip_stage1_rl",
        )
        self.skip_noise_rl = self._coerce_bool_flag(
            skip_noise_rl, "skip_noise_rl",
        )
        self.skip_final_eval = self._coerce_bool_flag(
            skip_final_eval, "skip_final_eval",
        )
        self.final_eval_only = self._coerce_bool_flag(
            final_eval_only, "final_eval_only",
        )
        self.needs_stage2_fixed_config = not self.skip_noise_rl
        self.resume_run_dir = str(resume_run_dir or "").strip()

        if self.final_eval_only:
            if (
                    not self.skip_stage1_rl
                    or not self.skip_noise_rl
                    or self.skip_final_eval
                    or not self.search_best_config_input_path
            ):
                raise ValueError(
                    "selected-config final evaluation requires both searches "
                    "skipped, final evaluation enabled, and search-best JSON"
                )
        elif not self.skip_final_eval:
            raise ValueError(
                "search and final evaluation are separate commands; search runs "
                "must skip final evaluation"
            )

        if self.skip_stage1_rl and (
                self.stage1_rl_episodes_specified
                or self.stage1_rl_episodes != stage1_config.PPO_MAX_EPISODES
        ):
            raise ValueError(
                "stage1_rl_episodes was set while Stage 1 is skipped"
            )
        if (
                not self.skip_stage1_rl
                and self.stage1_rl_episodes < stage1_config.PPO_UPDATE_INTERVAL
        ):
            raise ValueError(
                f"stage1_rl_episodes must be at least {stage1_config.PPO_UPDATE_INTERVAL}"
            )
        if (
                self.skip_noise_rl
                and not self.final_eval_only
                and (
                    self.stage2_rl_episodes_specified
                    or self.stage2_rl_episodes != 0
                )
        ):
            raise ValueError(
                "stage2_rl_episodes was set while Stage 2 is skipped"
            )

        if (
            (not self.skip_noise_rl)
            and str(blb_v3_search_backend or "ppo").strip().lower() == "ppo"
            and self.stage2_rl_episodes != 0
            and self.stage2_rl_episodes < stage1_config.PPO_UPDATE_INTERVAL
        ):
            raise ValueError(
                f"stage2_rl_episodes={self.stage2_rl_episodes} is too small. "
                f"It must be >= PPO_UPDATE_INTERVAL ({stage1_config.PPO_UPDATE_INTERVAL}) so Stage-2 PPO can update at least once."
            )


        self.blb_v3_inproc_rescale_optimizer_root = (
            None
            if blb_v3_inproc_rescale_optimizer_root in (None, "")
            else str(blb_v3_inproc_rescale_optimizer_root)
        )
        self.blb_v3_rollout_size = (
            None if blb_v3_rollout_size in (None, "") else int(blb_v3_rollout_size)
        )
        self.blb_v3_eval_interval = (
            None if blb_v3_eval_interval in (None, "") else int(blb_v3_eval_interval)
        )
        self.blb_v3_save_interval = (
            None if blb_v3_save_interval in (None, "") else int(blb_v3_save_interval)
        )
        self.blb_v3_calibrate_baseline_samples = (
            None
            if blb_v3_calibrate_baseline_samples in (None, "")
            else int(blb_v3_calibrate_baseline_samples)
        )
        self.blb_v3_seed = (
            None if blb_v3_seed in (None, "") else int(blb_v3_seed)
        )
        self.blb_v3_reward_devices = (
            "" if blb_v3_reward_devices is None else str(blb_v3_reward_devices)
        )
        self.stage1_rl_devices = (
            "" if stage1_rl_devices is None else str(stage1_rl_devices)
        )
        self.blb_v3_online_k_trials = max(1, int(blb_v3_online_k_trials))
        self.blb_v3_terminal_eval_batch_size = max(
            1, int(blb_v3_terminal_eval_batch_size)
        )
        self.blb_v3_promotion_validation_trials = max(
            1, int(blb_v3_promotion_validation_trials)
        )
        self.blb_v3_final_selection_top_n = max(
            1, int(blb_v3_final_selection_top_n)
        )
        self.blb_v3_final_selection_validation_trials = max(
            1, int(blb_v3_final_selection_validation_trials)
        )
        self.blb_v3_promotion_margin_window = max(
            0.0, float(blb_v3_promotion_margin_window)
        )
        self.blb_v3_baseline_groups = int(blb_v3_baseline_groups)
        self.blb_v3_baseline_trials_per_group = int(
            blb_v3_baseline_trials_per_group
        )
        self.blb_v3_constraint_bootstrap_samples = int(
            blb_v3_constraint_bootstrap_samples
        )
        if min(
            self.blb_v3_baseline_groups,
            self.blb_v3_baseline_trials_per_group,
            self.blb_v3_constraint_bootstrap_samples,
        ) <= 0:
            raise ValueError("robust baseline counts must be positive")
        for field_name, value in (
            (
                "blb_v3_online_constraint_probability",
                blb_v3_online_constraint_probability,
            ),
            (
                "blb_v3_promotion_constraint_probability",
                blb_v3_promotion_constraint_probability,
            ),
            (
                "blb_v3_final_constraint_probability",
                blb_v3_final_constraint_probability,
            ),
        ):
            probability = float(value)
            if not 0.0 < probability <= 1.0:
                raise ValueError(f"{field_name} must be in (0, 1]")
            setattr(self, field_name, probability)
        if not (
            self.blb_v3_online_constraint_probability
            <= self.blb_v3_promotion_constraint_probability
            <= self.blb_v3_final_constraint_probability
        ):
            raise ValueError(
                "constraint probabilities must satisfy online <= promotion <= final"
            )
        from rfr.search.comparators.common.stage2_core import (
            normalize_search_backend,
            validate_comparator_scientific_parameters,
        )

        self.blb_v3_search_backend = normalize_search_backend(
            blb_v3_search_backend
        )
        self.blb_v3_search_initial_design_size = int(
            blb_v3_search_initial_design_size
        )
        self.blb_v3_search_candidate_pool_size = int(
            blb_v3_search_candidate_pool_size
        )
        self.blb_v3_search_population_size = int(
            blb_v3_search_population_size
        )
        self.blb_v3_search_mutation_max_coordinates = int(
            blb_v3_search_mutation_max_coordinates
        )
        self.blb_v3_search_rf_n_estimators = int(
            blb_v3_search_rf_n_estimators
        )
        self.blb_v3_search_rf_min_samples_leaf = int(
            blb_v3_search_rf_min_samples_leaf
        )
        self.comparator_bo_stage1_no_improvement = self._coerce_positive_int(
            comparator_bo_stage1_no_improvement,
            "comparator_bo_stage1_no_improvement",
        )
        self.comparator_bo_stage2_no_improvement = self._coerce_positive_int(
            comparator_bo_stage2_no_improvement,
            "comparator_bo_stage2_no_improvement",
        )
        self.comparator_greedy_stage1_no_improvement_rounds = (
            self._coerce_positive_int(
                comparator_greedy_stage1_no_improvement_rounds,
                "comparator_greedy_stage1_no_improvement_rounds",
            )
        )
        self.comparator_greedy_stage2_no_improvement_rounds = (
            self._coerce_positive_int(
                comparator_greedy_stage2_no_improvement_rounds,
                "comparator_greedy_stage2_no_improvement_rounds",
            )
        )
        self.comparator_ga_stage1_generations = self._coerce_positive_int(
            comparator_ga_stage1_generations,
            "comparator_ga_stage1_generations",
        )
        self.comparator_ga_stage2_generations = self._coerce_positive_int(
            comparator_ga_stage2_generations,
            "comparator_ga_stage2_generations",
        )
        self.comparator_stage1_only = self._coerce_bool_flag(
            comparator_stage1_only, "comparator_stage1_only"
        )
        if self.blb_v3_search_backend == "ppo":
            if self.comparator_stage1_only:
                raise ValueError("comparator flags require a comparator backend")
        else:
            model_id = str(getattr(getattr(self.model, "config", None), "_name_or_path", ""))
            if (
                int(self.total_layers) != 12
                or str(self.data_path).lower() != "mrpc"
                or model_id.lower() != "textattack/bert-base-uncased-mrpc"
            ):
                raise ValueError(
                    "formal comparators require textattack BERT-base MRPC"
                )
            if int(self.random_seed) != 42 or int(self.blb_v3_seed or 42) != 42:
                raise ValueError("formal comparators require seed 42")
            if (
                int(self.stage2_inference_batch_size or self.batch_size)
                != MRPC_STAGE2_RL_ALIGNMENT_BATCH_SIZE
            ):
                raise ValueError(
                    "formal comparators require Stage-2 batch size "
                    f"{MRPC_STAGE2_RL_ALIGNMENT_BATCH_SIZE}"
                )
            validate_comparator_scientific_parameters(
                communication_importance_ratio=(
                    self.stage2_communication_importance_ratio
                ),
                truncation_backend="binary",
                truncation_ring_bits=43,
                truncation_source_fractional_bits=24,
            )
            calibration_samples = int(
                self.blb_v3_calibrate_baseline_samples or 8
            )
            evidence_contract = (
                calibration_samples,
                self.blb_v3_online_k_trials,
                self.blb_v3_constraint_bootstrap_samples,
                self.blb_v3_online_constraint_probability,
                self.blb_v3_promotion_constraint_probability,
                self.blb_v3_final_constraint_probability,
            )
            if evidence_contract != (8, 3, 4096, 0.50, 0.80, 0.95):
                raise ValueError("formal comparator evidence contract mismatch")
            if self.blb_v3_search_backend == "coinn_ga" and (
                self.blb_v3_search_population_size != 64
                or self.comparator_ga_stage1_generations <= 0
                or self.comparator_ga_stage2_generations <= 0
            ):
                raise ValueError("formal COINN-GA contract mismatch")
            if self.blb_v3_search_backend == "bo_rf" and (
                self.blb_v3_search_initial_design_size != 64
                or self.blb_v3_search_candidate_pool_size != 2_048
                or self.comparator_bo_stage1_no_improvement <= 0
                or self.comparator_bo_stage2_no_improvement <= 0
                or self.blb_v3_search_rf_n_estimators != 128
                or self.blb_v3_search_rf_min_samples_leaf != 2
            ):
                raise ValueError("formal BO-RF contract mismatch")
            if self.blb_v3_search_backend == "greedy" and (
                self.comparator_greedy_stage1_no_improvement_rounds <= 0
                or self.comparator_greedy_stage2_no_improvement_rounds <= 0
            ):
                raise ValueError("formal Greedy contract mismatch")
            if self.comparator_stage1_only:
                if self.skip_stage1_rl or not self.skip_noise_rl or not self.skip_final_eval:
                    raise ValueError("Stage-1-only comparator routing mismatch")
            elif self.skip_stage1_rl or self.skip_noise_rl:
                raise ValueError("two-stage comparator must run both searches")
            if self.blb_v3_final_selection_top_n != 5:
                raise ValueError("formal comparator strict top-5 contract mismatch")
            if self.blb_v3_search_mutation_max_coordinates != 4:
                raise ValueError("formal comparator mutation cap mismatch")
            trial_contract = (
                self.blb_v3_baseline_groups,
                self.blb_v3_baseline_trials_per_group,
                int(self.stage2_k_trials),
                self.blb_v3_promotion_validation_trials,
                self.blb_v3_final_selection_validation_trials,
            )
            if trial_contract != (5, 3, 3, 15, 15):
                raise ValueError("formal comparator trial contract mismatch")
            constraint_contract = (
                float(self.error_threshold),
                float(self.stage2_limit_tolerance),
                float(self.stage2_stability_tolerance),
                float(self.stage2_stability_multiplier),
                int(self.stage2_probe_size),
            )
            if not (
                math.isclose(constraint_contract[0], 0.001)
                and math.isclose(constraint_contract[1], 0.001)
                and math.isclose(constraint_contract[2], 1.2)
                and math.isclose(constraint_contract[3], 2.0)
                and constraint_contract[4] == 256
            ):
                raise ValueError("formal comparator constraint contract mismatch")


    @staticmethod
    def _coerce_bool_flag(raw_value, flag_name):
        if isinstance(raw_value, bool):
            return raw_value
        if raw_value is None:
            return False

        text = str(raw_value).strip().lower()
        if text in ('1', 'true', 't', 'yes', 'y', 'on'):
            return True
        if text in ('0', 'false', 'f', 'no', 'n', 'off', ''):
            return False

        raise ValueError(
            f"Invalid boolean value for {flag_name}: {raw_value!r}. "
            "Expected one of: true/false/1/0/yes/no."
        )

    @staticmethod
    def _coerce_positive_int(raw_value, flag_name):
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

    @staticmethod
    def _coerce_nonnegative_int(raw_value, flag_name):
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid nonnegative integer for {flag_name}: {raw_value!r}."
            ) from None

        if value < 0:
            raise ValueError(
                f"Invalid nonnegative integer for {flag_name}: {raw_value!r}."
            )
        return value


    def _write_stage1_best_config_file(
            self,
            *,
            best_config: Mapping[str, Any],
            completed_episodes: int,
            ) -> str:
        """Write the sole production handoff consumed by Stage 2."""
        from rfr.search.common.best_config import write_stage1_best_config

        if not self.run_output_dir:
            raise RuntimeError("Stage-1 best-config export requires run_output_dir")
        algorithm = (
            "rl" if self.blb_v3_search_backend == "ppo"
            else self.blb_v3_search_backend
        )
        path = write_stage1_best_config(
            self.run_output_dir,
            algorithm=algorithm,
            model_type=(
                "bert-large" if int(self.total_layers) == 24 else "bert-base"
            ),
            dataset=self.dataset_key,
            gelu=best_config["gelu"],
            softmax=best_config["softmax"],
            selection={
                "status": str(
                    best_config.get("selection_status") or "selected"
                ),
                "feasible": bool(best_config.get("feasible", True)),
                "evaluation": best_config.get("evaluation"),
            },
            provenance={
                "dataset_protocol_hash": self.dataset_protocol_hash,
                "completed_evaluations": int(completed_episodes),
                "source_result_path": best_config.get("result_path"),
                "source_result_sha256": best_config.get("result_sha256"),
                "search_accounting": best_config.get("search_accounting"),
            },
        )
        self.stage1_best_config_path = str(path)
        self.log(f"  Stage-1 selected configuration: {path}")
        return str(path)

    def _write_search_best_config_file(
            self,
            *,
            stage1_config: Mapping[str, Any],
            stage2_result: Mapping[str, Any],
            ) -> str:
        """Write the sole selected-configuration input accepted by final eval."""
        from rfr.search.common.best_config import write_search_best_config

        if not self.run_output_dir:
            raise RuntimeError("search best-config export requires run_output_dir")
        group = stage2_result.get("blb_v3_best_action_group")
        if not isinstance(group, Mapping):
            raise RuntimeError("completed Stage-2 result has no action group")
        action_matrix = group.get("policy_actions")
        if not isinstance(action_matrix, (list, tuple)):
            raise RuntimeError("completed Stage-2 result has no action matrix")
        status = str(stage2_result.get("status") or "")
        if status not in ("completed", "completed_infeasible"):
            raise RuntimeError(
                f"Stage-2 status {status!r} cannot produce a formal best config"
            )
        algorithm = (
            "rl" if self.blb_v3_search_backend == "ppo"
            else self.blb_v3_search_backend
        )
        strict_feasible = bool(
            stage2_result.get("strict_feasible", status == "completed")
        )
        path = write_search_best_config(
            self.run_output_dir,
            algorithm=algorithm,
            model_type=(
                "bert-large" if int(self.total_layers) == 24 else "bert-base"
            ),
            dataset=self.dataset_key,
            gelu=stage1_config["gelu"],
            softmax=stage1_config["softmax"],
            action_matrix=action_matrix,
            selection={
                "status": status,
                "strict_feasible": strict_feasible,
                "final_eval_eligible": strict_feasible,
                "scientific_status": stage2_result.get("scientific_status"),
                "final_config_fingerprint": stage2_result.get(
                    "final_config_fingerprint"
                ),
            },
            provenance={
                "dataset_protocol_hash": self.dataset_protocol_hash,
                "stage1_config_path": self.stage1_best_config_input_path,
                "stage1_config_sha256": getattr(
                    self, "stage1_best_config_input_sha256", None
                ),
                "search_accounting": stage2_result.get("search_accounting"),
                "algorithm_contract_hash": stage2_result.get(
                    "algorithm_contract_hash"
                ),
                "run_context_hash": stage2_result.get("run_context_hash"),
            },
        )
        self.search_best_config_path = str(path)
        self.log(f"  Selected two-stage configuration: {path}")
        return str(path)

    def _resolve_stage2_fixed_stage1_config(self):
        """Load the sole Stage-1 JSON handoff accepted by Stage 2."""
        from rfr.search.common.best_config import load_stage1_best_config

        generated_path = str(getattr(self, "stage1_best_config_path", "") or "")
        path = generated_path or self.stage1_best_config_input_path
        if not path:
            raise ValueError("Stage 2 requires a Stage-1 selected-config JSON")
        payload = load_stage1_best_config(path)
        expected_algorithm = (
            "rl" if self.blb_v3_search_backend == "ppo"
            else self.blb_v3_search_backend
        )
        expected_model = (
            "bert-large" if int(self.total_layers) == 24 else "bert-base"
        )
        if payload["algorithm"] != expected_algorithm:
            raise ValueError(
                "Stage-1 selected-config algorithm does not match Stage 2"
            )
        if payload["model_type"] != expected_model:
            raise ValueError(
                "Stage-1 selected-config model does not match Stage 2"
            )
        if payload["dataset"] != self.dataset_key:
            raise ValueError(
                "Stage-1 selected-config dataset does not match Stage 2"
            )
        gelu = np.asarray(payload["stage1"]["gelu"], dtype=int)
        softmax = np.asarray(payload["stage1"]["softmax"], dtype=int)
        with open(path, "rb") as handle:
            content_hash = hashlib.sha256(handle.read()).hexdigest()
        self.stage1_best_config_input_path = os.path.abspath(path)
        self.stage1_best_config_input_sha256 = content_hash
        self.stage1_best_config_payload = payload
        label = f"Stage-1 selected config ({expected_algorithm})"
        source = f"stage1_json:{self.stage1_best_config_input_path}"
        return gelu, softmax, label, source

    def _detect_task_type(self):
        self.dataset_key = validate_dataset(self.data_path)
        self.dataset_config = DATASET_METRICS_CONFIG[self.dataset_key]
        return False

    def _log_task_type(self):
        """Log whether the task is classification or regression."""
        full_names = self.dataset_config['metric_full_names']
        task_type = 'REGRESSION' if self.is_regression else 'CLASSIFICATION'
        print(f"[信息] 数据集（Dataset）'{self.data_path}' 检测为 {task_type} 任务")
        print(f"[信息] 使用指标（Using metrics）: {', '.join(full_names)}")

    def get_metric_names(self) -> Tuple[str, ...]:
        """Return metric names for the active GLUE task."""
        full_names = self.dataset_config['metric_full_names']
        if len(full_names) == 1:
            return (full_names[0],)
        return (full_names[0], full_names[1])

    def get_metric_short_names(self) -> Tuple[str, ...]:
        """Return abbreviated metric names for logs and reports."""
        return self.dataset_config['metric_names']

    def get_num_metrics(self) -> int:
        """Return the number of task metrics."""
        return len(self.dataset_config['metrics'])

    def _make_dataloader(self, dataset, *, batch_size=None):
        if dataset is None:
            return None
        effective_batch_size = (
            self._active_inference_batch_size
            if batch_size is None else max(1, int(batch_size))
        )

        return DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            pin_memory=torch.cuda.is_available(),
        )

    def activate_stage2_inference_batch_size(self):
        """Switch every evaluator loader to the Stage-2 scientific batch."""
        target = (
            self.batch_size
            if self.stage2_inference_batch_size is None
            else int(self.stage2_inference_batch_size)
        )
        if target <= 0:
            raise ValueError("stage2_inference_batch_size must be positive")
        if int(self._active_inference_batch_size) == target:
            return target

        self._active_inference_batch_size = target
        rebuilt = {}
        for split_name, dataset in self.dataset_splits.items():
            loader = self._make_dataloader(dataset, batch_size=target)
            rebuilt[split_name] = (
                tuple(loader) if split_name == "validation_full" else loader
            )
        self.dataloaders = rebuilt
        self.dataloader_train = self.dataloaders.get("train")
        self.dataloader_test = self.dataloaders.get("validation_full")


        from rfr.search.rl.stage1.eval_cache import Stage1EvalCache

        self._eval_cache = Stage1EvalCache()
        self._stage1_worker_eval_cache = Stage1EvalCache()
        return target

    def _register_dataset_split(self, split_name, dataset):
        if dataset is None:
            self.dataset_splits.pop(split_name, None)
            self.dataloaders.pop(split_name, None)
            return

        cache_eval_batches = split_name == "validation_full"
        self.dataset_splits[split_name] = dataset
        dataloader = self._make_dataloader(dataset)
        self.dataloaders[split_name] = tuple(dataloader) if cache_eval_batches else dataloader

    def has_dataset_split(self, split_name):
        return self.dataloaders.get(split_name) is not None


    def _prepare_rl_datasets(self, train_data, train_probe, validation_data):
        self.dataset_splits = {}
        self.dataloaders = {}

        self._register_dataset_split("train", train_data)
        self._register_dataset_split("train_probe", train_probe)
        self._register_dataset_split("validation_full", validation_data)

        self.dataloader_train = self.dataloaders.get("train")
        self.dataloader_test = self.dataloaders.get("validation_full")

    def get_reward_reference_split_name(self):
        return TRAIN_PROBE_SPLIT

    def get_online_reward_split_name(self):
        return TRAIN_PROBE_SPLIT

    def _resolve_eval_split(self, use_train=True, split=None):
        if split is not None:
            if not self.has_dataset_split(split):
                raise ValueError(f"Unknown or unavailable dataset split: {split}")
            return split

        if use_train:
            return "train"
        if self.has_dataset_split("validation_full"):
            return "validation_full"
        return "train"

    def _candidate_meets_constraints(self, loss, metric1, metric2, limit_loss, limit_p, limit_s):
        if loss > limit_loss:
            return False
        if metric1 < limit_p:
            return False
        if self.get_num_metrics() > 1 and metric2 < limit_s:
            return False
        return True

    def _is_better_confirmed_candidate(self, candidate, incumbent, metric_prefix):
        if candidate is None:
            return False
        if incumbent is None:
            return True

        cand_metric_sum = (
            float(candidate.get(f"{metric_prefix}_metric1", -float("inf"))) +
            float(candidate.get(f"{metric_prefix}_metric2", -float("inf")))
        )
        inc_metric_sum = (
            float(incumbent.get(f"{metric_prefix}_metric1", -float("inf"))) +
            float(incumbent.get(f"{metric_prefix}_metric2", -float("inf")))
        )
        if cand_metric_sum > inc_metric_sum + 1e-8:
            return True
        if cand_metric_sum < inc_metric_sum - 1e-8:
            return False

        cand_loss = float(candidate.get(f"{metric_prefix}_loss", float("inf")))
        inc_loss = float(incumbent.get(f"{metric_prefix}_loss", float("inf")))
        if cand_loss < inc_loss - 1e-8:
            return True
        if cand_loss > inc_loss + 1e-8:
            return False

        cand_cost = float(candidate["cost"])
        inc_cost = float(incumbent["cost"])
        if cand_cost < inc_cost - 1e-8:
            return True
        if cand_cost > inc_cost + 1e-8:
            return False

        return float(candidate.get("proxy_reward", -float("inf"))) > float(
            incumbent.get("proxy_reward", -float("inf"))
        )

    @staticmethod
    def _select_stage1_reward_best_config(
            best_config,
            best_reward,
            base_gelu,
            base_softmax,
            base_tot_c,
            ):
        if best_config is None or best_reward < -50:
            return {
                'gelu': base_gelu.copy(),
                'softmax': base_softmax.copy(),
                'cost': base_tot_c,
                'reward': 0,
            }, True
        return {
            k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in best_config.items()
        }, False

    def _fmt_metrics(self, loss, m1, m2, prefix=""):
        """Format the current metrics for a progress line."""
        names = self.get_metric_names()
        p = f"{prefix}" if prefix else ""
        if self.get_num_metrics() == 1:
            return f"{p}损失（Loss）: {loss:.6f}, {names[0]}: {m1:.6f}"
        return f"{p}损失（Loss）: {loss:.6f}, {names[0]}: {m1:.6f}, {names[1]}: {m2:.6f}"

    def _fmt_constraints(self, limit_loss, limit_p, limit_s):
        """Format constraint values for a progress line."""
        names = self.get_metric_names()
        if self.get_num_metrics() == 1:
            return f"损失（Loss）<={limit_loss:.4f}, {names[0]}>={limit_p:.4f}"
        return f"损失（Loss）<={limit_loss:.4f}, {names[0]}>={limit_p:.4f}, {names[1]}>={limit_s:.4f}"

    def _write_step_info(self, step_info, f):
        """Write one Stage 1 search step to the log."""
        f.write(f"  全局步数（step_global）: {step_info['step_global']}\n")
        f.write(f"  回合编号（episode_id）: {step_info['episode_id']}\n")
        f.write(f"  层索引（layer_index）: {step_info['layer_index']}\n")
        f.write(f"  状态向量（state_vector）: {step_info['state_vector']}\n")
        f.write(f"  当前GELU阶数（curr_gelu_degree）: {step_info['curr_gelu_degree']}\n")
        f.write(f"  当前Softmax阶数（curr_softmax_degree）: {step_info['curr_softmax_degree']}\n")
        f.write(f"  GELU概率分布（gelu_prob_dist）: {step_info['gelu_prob_dist']}\n")

        f.write(f"  评论家值（critic_value）: {step_info['critic_value']}\n")
        f.write(f"  累计成本（accumulated_cost）: {step_info['accumulated_cost']}\n")
        f.write(f"  GELU配置（gelu_config）: {step_info['gelu_config']}\n")
        f.write(f"  Softmax配置（softmax_config）: {step_info['softmax_config']}\n")

        if 'current_lr' in step_info:
            f.write(f"  当前学习率（current_lr）: {step_info['current_lr']:.6f}\n")
        if 'current_entropy_coef' in step_info:
            f.write(f"  当前熵系数（current_entropy_coef）: {step_info['current_entropy_coef']:.6f}\n")

    def update_hyperparameters(self, optimizer, episode):
        """Update the production entropy schedule and report the optimizer LR."""
        self.current_episode = episode
        progress = episode / self.total_episodes

        plateau = 0.25
        if progress <= plateau:
            new_entropy = stage1_config.PPO_ENTROPY_START
        else:
            tail = (progress - plateau) / max(1.0 - plateau, 1e-8)
            tail = min(max(tail, 0.0), 1.0)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * tail))
            new_entropy = stage1_config.PPO_ENTROPY_END + (
                stage1_config.PPO_ENTROPY_START - stage1_config.PPO_ENTROPY_END
            ) * cosine_factor
        new_entropy = max(new_entropy, stage1_config.GTRXL_ENTROPY_LOWER_BOUND)
        self.current_entropy_coef = new_entropy


        if hasattr(optimizer, 'param_groups'):
            self.current_lr = optimizer.param_groups[0]['lr']
        elif isinstance(optimizer, dict):
            self.current_lr = optimizer.get('actor', optimizer.get('critic', {}))
            if hasattr(self.current_lr, 'param_groups'):
                self.current_lr = self.current_lr.param_groups[0]['lr']


        return self.current_lr, new_entropy

    def _rebuild_reward_statistics_accumulators(self):
        self.reward_history_sum = float(sum(float(value) for value in self.reward_history))
        self.reward_history_sumsq = float(sum(float(value) * float(value) for value in self.reward_history))

    def get_current_entropy_coef(self):
        """Return the current entropy coefficient."""
        return self.current_entropy_coef

    def update_reward_statistics(self, episode_reward):
        """Update online reward normalization statistics."""
        episode_reward = float(episode_reward)
        if len(self.reward_history) == self.reward_history.maxlen:
            old_reward = float(self.reward_history[0])
            self.reward_history_sum -= old_reward
            self.reward_history_sumsq -= old_reward * old_reward
        self.reward_history.append(episode_reward)
        self.reward_history_sum += episode_reward
        self.reward_history_sumsq += episode_reward * episode_reward


        if len(self.reward_history) >= stage1_config.RUNNING_REWARD_MIN_SAMPLES:
            n = float(len(self.reward_history))
            self.reward_mean = self.reward_history_sum / n
            variance = max(self.reward_history_sumsq / n - self.reward_mean * self.reward_mean, 0.0)
            self.reward_std = math.sqrt(variance) + stage1_config.RUNNING_REWARD_EPSILON

    def _detect_layer_attribute(self):
        layers = getattr(getattr(self.model, "bert", None), "encoder", None)
        layers = getattr(layers, "layer", None)
        if layers is None or len(layers) == 0:
            raise TypeError("SearchWorkflow requires a BERT encoder")
        return 'bert.encoder.layer'

    def _get_layer_act_module(self, layer):
        """Return a BERT layer's activation module when it is replaceable."""
        inter = getattr(layer, "intermediate", None)
        if inter is not None and hasattr(inter, "intermediate_act_fn"):
            act = inter.intermediate_act_fn
            return act if isinstance(act, nn.Module) else None
        return None

    def log(self, message):
        print(message, flush=True)
        target_log_file = getattr(self, "active_log_file", self.log_file)
        ensure_parent_dir(target_log_file)
        with open(target_log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def _initialize_noise_log_file(self):
        if getattr(self, "_noise_log_initialized", False):
            return
        if self.noise_log_file == self.log_file:
            self._noise_log_initialized = True
            return

        ensure_parent_dir(self.noise_log_file)

        _is_resuming = bool(getattr(self, "resume_run_dir", ""))
        _noise_log_mode = "a" if _is_resuming and os.path.isfile(self.noise_log_file) else "w"
        _lr_mode_labels = {
            "direct": "直接指定（direct）",
            "default": "默认值（default）",
        }
        _stage1_lr_mode = _lr_mode_labels.get(str(self.stage1_ppo_lr_mode), str(self.stage1_ppo_lr_mode))
        _stage2_lr_mode = _lr_mode_labels.get(str(self.stage2_ppo_lr_mode), str(self.stage2_ppo_lr_mode))
        with open(self.noise_log_file, _noise_log_mode, encoding="utf-8") as f:


            if _is_resuming and _noise_log_mode == "a":
                import datetime as _dt
                header_lines = [
                    "",
                    "=" * 80,
                    f"【二阶段噪声 RL 续训日志】时间：{_dt.datetime.now().isoformat()}",
                    "=" * 80,
                    "",
                ]
            else:
                header_lines = [
                    "=" * 80,
                    "【二阶段噪声 RL 日志】二阶段噪声 RL 日志开始（Stage-2 noise RL log started）",
                    "=" * 80,
                    "",
                ]
            f.write("\n".join(header_lines))
        with open(self.noise_log_file, "a", encoding="utf-8") as f:
            f.write(
                "【学习率配置】\n"
                f"  - 一阶段 PPO 学习率（Stage-1 PPO LR）：raw={self.stage1_rl_lr_raw!r} -> "
                f"{self.stage1_ppo_lr_initial:.6g}（{_stage1_lr_mode}）\n"
                f"  - 二阶段 PPO 学习率（Stage-2 PPO LR）：raw={self.stage2_rl_lr_raw!r} -> "
                f"{self.stage2_ppo_lr_initial:.6g}（{_stage2_lr_mode}）\n"
            )
            f.write(
                "【训练轮数配置】\n"
                f"  - 一阶段 RL 回合数（stage-1 episodes）：{self.stage1_rl_episodes}\n"
                f"  - 二阶段噪声 RL 回合数（stage-2 episodes）：{self.stage2_rl_episodes}\n"
            )
            if self.run_output_dir:
                f.write(
                    "【输出目录】\n"
                    f"  - 本次运行根目录（run_output_dir）：{self.run_output_dir}\n"
                )
        self._noise_log_initialized = True

    def activate_noise_logging(self):
        previous_log_file = getattr(self, "active_log_file", self.log_file)
        self._initialize_noise_log_file()
        self.active_log_file = self.noise_log_file
        return previous_log_file

    def restore_log_file(self, previous_log_file):
        self.active_log_file = previous_log_file or self.log_file

    def get_simulated_cost(
            self,
            gelu_degrees: Sequence[int],
            softmax_degrees: Sequence[int],
    ) -> Tuple[float, float, float]:
        """Return ``(total_cost, gelu_cost, softmax_cost)`` proxy cost for a
        Stage-1 GELU/Softmax polynomial degree assignment."""
        g_c = sum(self.GELU_COST_MAP.get(int(d), 0) for d in gelu_degrees)
        s_c = sum(self.SOFTMAX_COST_MAP.get(int(d), 0) for d in softmax_degrees)
        return g_c + s_c, g_c, s_c

    def apply_configuration(
            self,
            gelu_degrees: Sequence[int],
            softmax_degrees: Sequence[int],
    ) -> None:
        """Install GELU + Softmax polynomial approximations on every layer of
        the underlying transformer. Eagerly switches the model to ``eval()``
        because candidate scoring is always inference. Mutates the model
        in-place; no return value."""


        model = getattr(self, "model", None)
        if model is None:
            model = getattr(getattr(self, "reversible_handler", None), "model", None)
        if model is not None:
            model.eval()
        cfg_sig = (
            tuple(int(d) for d in gelu_degrees),
            tuple(int(d) for d in softmax_degrees),
        )
        previous_cfg = getattr(self, "_last_applied_config", None)
        if previous_cfg == cfg_sig:
            return
        handler_layer_name = "model." + self.layers_attribute
        self._last_applied_config = None
        try:
            self.reversible_handler._last_stage1_applied_config = None
        except Exception:
            pass
        _install_stage1_function_configuration(
            self.reversible_handler,
            handler_layer_name,
            cfg_sig,
            previous_cfg=previous_cfg,
        )
        self._last_applied_config = cfg_sig
        try:
            self.reversible_handler._last_stage1_applied_config = cfg_sig
        except Exception:
            pass

    @staticmethod
    def _unsupported_int_values(values, allowed_values):
        allowed_set = {int(value) for value in allowed_values}
        invalid = set()
        for value in np.asarray(values, dtype=int).reshape(-1):
            int_value = int(value)
            if int_value not in allowed_set:
                invalid.add(int_value)
        return sorted(invalid)

    def stage1_evaluate(self, gelu_degrees, softmax_degrees, use_train=True, split=None):
        """Stage-1 scoring is plaintext-only: replace GELU/Softmax, no BLB noise."""
        return self.evaluate_model(
            gelu_degrees, softmax_degrees, use_train=use_train, split=split,
        )

    def _stage1_evaluate_on_model(self, *, model, handler, device,
                                   gelu_degrees, softmax_degrees, split_name):
        """Stateless variant of ``stage1_evaluate`` for an explicit (model,
        handler, device) triple. Used by the Stage-1 multi-GPU rollout
        runner so worker N can evaluate on its replica without touching
        ``self.model`` / ``self.reversible_handler`` / ``self.device``.

        Differences vs ``stage1_evaluate``:
          * Uses the lock-protected worker cache instead of the single-GPU
            cache, so worker writes remain thread-safe.
          * Does NOT write to ``self.current_*_scaling_factors`` state
            attributes (race risk under concurrent workers).
          * GELU/Softmax install is inlined against the passed ``handler``.
            Stage-1 worker scoring is plaintext-only and intentionally never
            installs Stage-2 noise hooks.
          * The forward loop reuses ``_run_evaluation`` via its
            ``model`` / ``device`` overrides.

        Returns the same ``(loss, metric1, metric2, time_ms)`` tuple.
        """
        if handler is None or model is None or device is None:
            raise ValueError("_stage1_evaluate_on_model requires explicit model/handler/device")
        if split_name is None:
            raise ValueError("_stage1_evaluate_on_model requires explicit split_name")


        _shared_cache = getattr(self, "_stage1_worker_eval_cache", None)
        if _shared_cache is not None:
            _cache_key = _shared_cache.make_key(gelu_degrees, softmax_degrees, split_name)
            _cached = _shared_cache.get(_cache_key)
            if _cached is not None:
                return _cached

        handler_layer_name = "model." + self.layers_attribute

        model.eval()
        cfg_sig = (
            tuple(int(d) for d in gelu_degrees),
            tuple(int(d) for d in softmax_degrees),
        )
        previous_cfg = getattr(handler, "_last_stage1_applied_config", None)
        if previous_cfg != cfg_sig:
            handler._last_stage1_applied_config = None
            _install_stage1_function_configuration(
                handler,
                handler_layer_name,
                cfg_sig,
                previous_cfg=previous_cfg,
            )
            handler._last_stage1_applied_config = cfg_sig


        dataloader = self.dataloaders[split_name]
        _forward_t0 = time.time()
        result = self._run_evaluation(
            dataloader,
            use_train=(split_name == "train"),
            split_name=split_name,
            model=model,
            device=device,
        )
        self._record_stage1_parallel_model_forward(time.time() - _forward_t0)
        if _shared_cache is not None:
            _shared_cache.put(_cache_key, result)
        return result

    def _record_stage1_parallel_model_forward(self, seconds):
        lock = self._stage1_parallel_timing_state_lock()
        with lock:
            self._stage1_parallel_model_forward_seconds += max(0.0, float(seconds))
            self._stage1_parallel_model_forward_calls += 1

    def _stage1_worker_timing_snapshot(self):
        lock = self._stage1_parallel_timing_state_lock()
        with lock:
            return {
                "model_forward_seconds": float(self._stage1_parallel_model_forward_seconds),
                "model_forward_calls": int(self._stage1_parallel_model_forward_calls),
            }

    def _stage1_parallel_timing_state_lock(self):
        lock = getattr(self, "_stage1_parallel_timing_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._stage1_parallel_timing_lock = lock
            self._stage1_parallel_model_forward_seconds = float(
                getattr(self, "_stage1_parallel_model_forward_seconds", 0.0)
            )
            self._stage1_parallel_model_forward_calls = int(
                getattr(self, "_stage1_parallel_model_forward_calls", 0)
            )
        return lock

    def stage1_final_evaluate(self, gelu_degrees, softmax_degrees, use_train=False, split=None):
        """Stage-1 final eval is plaintext-only: GELU/Softmax replacement, no BLB noise."""
        return self.evaluate_model(
            gelu_degrees, softmax_degrees, use_train=use_train, split=split,
        )

    def build_constraint_limits_from_metrics(
            self,
            base_loss,
            base_p,
            base_s,
            error_threshold=None,
            correlation_drop_ratio=None,
            ):
        error_threshold = (
            self.error_threshold
            if error_threshold is None
            else float(error_threshold)
        )
        correlation_drop_ratio = (
            self.correlation_drop_ratio
            if correlation_drop_ratio is None
            else float(correlation_drop_ratio)
        )
        return {
            "loss": float(base_loss * (1.0 + error_threshold)),
            "metric1": float(base_p * (1.0 - correlation_drop_ratio)),
            "metric2": float(base_s * (1.0 - correlation_drop_ratio)),
        }

    def get_stage1_exact_baseline_configuration(self):
        return (
            np.full(self.total_layers, stage1_config.STAGE1_ORIGINAL_FUNCTION_DEGREE, dtype=int),
            np.full(self.total_layers, stage1_config.STAGE1_ORIGINAL_FUNCTION_DEGREE, dtype=int),
        )

    def get_stage1_cost_reference_configuration(self):
        return (
            np.full(self.total_layers, 4, dtype=int),
            np.full(self.total_layers, 6, dtype=int),
        )

    def evaluate_model_repeated(
            self,
            gelu_degrees,
            softmax_degrees,
            repeats=1,
            use_train=True,
            split=None,
            ):
        repeats = max(1, int(repeats))
        split_name = self._resolve_eval_split(use_train=use_train, split=split)
        trials = []
        for _ in range(repeats):
            loss, p, s, t = self.evaluate_model(
                gelu_degrees,
                softmax_degrees,
                use_train=(split_name == "train"),
                split=split_name,
            )
            trials.append(
                {
                    "loss": float(loss),
                    "p": float(p),
                    "s": float(s),
                    "time_ms": float(t),
                }
            )

        summary = summarize_eval_trials(trials)
        summary.update({
            "split_name": split_name,
            "trials": trials,
        })
        return summary

    def _reset_runtime_ppo_state(self, stage_label='stage1'):
        self.current_episode = 0
        self.current_entropy_coef = stage1_config.PPO_ENTROPY_INITIAL
        if stage_label == 'stage2':
            self.current_lr = self.stage2_ppo_lr_initial
        else:
            self.current_lr = self.stage1_ppo_lr_initial
        self.reward_history = deque(maxlen=stage1_config.RUNNING_REWARD_HISTORY_SIZE)
        self.reward_history_sum = 0.0
        self.reward_history_sumsq = 0.0
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.return_normalizer = RunningMeanStd()

    def _get_stage1_resume_checkpoint_path(self):
        """Resolve the checkpoint path used for Stage 1 resume."""
        from rfr.search.rl.stage1.checkpoint import STAGE1_CHECKPOINT_FILENAME
        if not self.resume_run_dir:
            return None
        path = os.path.join(self.resume_run_dir, "stage1", STAGE1_CHECKPOINT_FILENAME)
        return path if os.path.isfile(path) else None

    def _get_stage2_resume_checkpoint_path(self):
        if not self.resume_run_dir:
            return None
        from rfr.search.rl.stage2.entrypoint import (
            BLB_STAGE2_FINAL_CHECKPOINT_FILENAME,
            BLB_STAGE2_LIVE_CHECKPOINT_FILENAME,
        )

        for filename in (
            BLB_STAGE2_FINAL_CHECKPOINT_FILENAME,
            BLB_STAGE2_LIVE_CHECKPOINT_FILENAME,
        ):
            for progress_dir_name in ("stage2",):
                path = os.path.join(
                    self.resume_run_dir,
                    progress_dir_name,
                    "progress",
                    filename,
                )
                if os.path.isfile(path):
                    return path
        return None

    def run_noise_rl_stage(
        self,
        fixed_gelu,
        fixed_softmax,
        fixed_label,
        fixed_source,
        resume_checkpoint_path=None,
    ):
        from rfr.search.rl.stage2 import BLBStage2RLRunner

        runner = BLBStage2RLRunner(self)
        return runner.run(
            fixed_gelu,
            fixed_softmax,
            fixed_label,
            fixed_source,
            resume_checkpoint_path=resume_checkpoint_path,
        )

    def save_best_policies_snapshot(self):
        """Persist the current best policies without changing training state."""
        import shutil
        bp_dir = os.path.join(self.run_output_dir, "best_policy")
        os.makedirs(bp_dir, exist_ok=True)
        _candidates = [
            (os.path.join(os.path.dirname(self.step_info_file), "stage1_policy.pt"), "stage1_policy.pt"),
            (os.path.join(self.noise_stage_progress_dir, "stage2_noise_policy.pt"), "stage2_noise_policy.pt"),
        ]
        for src, dst_name in _candidates:
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(bp_dir, dst_name))
                self.log(f"  [best_policy] 已复制 {src} → {os.path.join(bp_dir, dst_name)}")

        import json as _json
        meta = {
            "stage1_accuracy_tolerance": float(self.error_threshold),
            "stage2_limit_tolerance": float(self.stage2_limit_tolerance),
            "stage2_stability_tolerance": float(self.stage2_stability_tolerance),
            "stage2_communication_importance_ratio": float(
                self.stage2_communication_importance_ratio
            ),
            "stage2_k_trials": int(self.stage2_k_trials),
            "stage2_probe_size": int(self.stage2_probe_size),
            "stage1_inference_batch_size": int(self.batch_size),
            "stage2_inference_batch_size": int(
                getattr(self, "_active_inference_batch_size", self.batch_size)
            ),
            "dataset": str(getattr(self, "data_path", "")),
            "search_algorithm": str(getattr(self, "search_algorithm", "")),
        }
        with open(os.path.join(bp_dir, "constraint_metadata.json"), "w") as _f:
            _json.dump(meta, _f, indent=2)

    def run_selected_config_final_eval(
            self,
            *,
            search_config,
            ):
        """Evaluate the one configuration loaded from search-best JSON."""
        from rfr.evaluation.action_eval import BLBActionFinalEvaluationModule

        runner = BLBActionFinalEvaluationModule(
            evaluator=self,
            random_seed=self.random_seed,
            repeat_n=self.final_eval_repeat_n,
            results_dir=self.final_eval_dir,
        )
        return runner.run(search_config=search_config)

    def _run_evaluation(self, dataloader, use_train=False, split_name=None, *,
                        model=None, device=None):
        """Evaluate the current model configuration and return task metrics."""


        _model = self.model if model is None else model
        _device = self.device if device is None else device


        _model.eval()


        if _model is self.model and not self._eval_infra_ready:
            _model.to(_device)
            self._eval_infra_ready = True
        ds = self.dataset_key
        try:
            result = run_installed_model_on_dataloader(
                _model,
                dataloader,
                device=_device,
                metric_profile=ds,
                use_train=bool(use_train),
                split_name=split_name,
                loss_average="batch",
            )
            avg_loss = result.loss
            metric1, metric2 = self._stage1_metric_pair_from_eval_result(result)
            avg_time = result.time_ms
        except Exception as e:
            raise RuntimeError(
                f"数据集(dataset){ds!r} 的模型评估失败。禁止将基础设施错误"
                "转换为零指标或写入评估缓存"
            ) from e
        return avg_loss, metric1, metric2, avg_time

    def evaluate_model(
            self,
            gelu_degrees: Sequence[int],
            softmax_degrees: Sequence[int],
            use_train: bool = True,
            split: Optional[str] = None,
    ) -> Tuple[float, float, float, float]:
        """Evaluate the active model on the configured data loader."""
        split_name = self._resolve_eval_split(use_train=use_train, split=split)
        cache_key = self._eval_cache.make_key(
            gelu_degrees,
            softmax_degrees,
            split_name,
        )
        cached = self._eval_cache.get(cache_key)
        if cached is not None:
            return cached

        cfg_sig = (
            tuple(int(d) for d in gelu_degrees),
            tuple(int(d) for d in softmax_degrees),
        )
        if self._last_applied_config != cfg_sig:
            self.apply_configuration(gelu_degrees, softmax_degrees)
            self._last_applied_config = cfg_sig
        dataloader = self.dataloaders[split_name]
        result = self._run_evaluation(
            dataloader,
            use_train=(split_name == "train"),
            split_name=split_name,
        )
        self._eval_cache.put(cache_key, result)
        return result

    @staticmethod
    def _logits_to_classes(all_preds):
        """Convert logits to task predictions."""
        preds_arr = np.array(all_preds)
        if len(preds_arr.shape) == 1:
            return (preds_arr > 0.5).astype(int)
        return np.argmax(preds_arr, axis=1)

    @staticmethod
    def _normalize_labels_for_metrics(labels):
        return shared_normalize_labels_for_metrics(labels)

    @staticmethod
    def _normalize_logits_for_metrics(logits, expected_batch_size):
        return shared_normalize_logits_for_metrics(logits, expected_batch_size)

    def _stage1_metric_pair_from_eval_result(self, result):
        pred_classes = self._logits_to_classes(result.logits)
        labels = result.labels
        metric1 = accuracy_score(labels, pred_classes)
        ds = str(getattr(self, "dataset_key", "") or "").lower()
        metric2 = (
            f1_score(labels, pred_classes, average="weighted")
            if ds == "mrpc"
            else metric1
        )
        return float(metric1), float(metric2)



    def on_evaluate(self, args, state, control, **kwargs):
        """Run Stage 1, then the configured Stage-2 or final-evaluation path."""
        best_config, base_gelu, base_softmax = run_stage1(self)

        noise_stage_result = None
        final_eval_result = None
        stage2_fixed_gelu = np.asarray(base_gelu, dtype=int)
        stage2_fixed_softmax = np.asarray(base_softmax, dtype=int)
        stage2_fixed_label = "Baseline"
        stage2_fixed_source = "baseline"
        if self.needs_stage2_fixed_config:
            (
                stage2_fixed_gelu,
                stage2_fixed_softmax,
                stage2_fixed_label,
                stage2_fixed_source,
            ) = self._resolve_stage2_fixed_stage1_config()
            self.log(
                f"[Info] Stage-2 固定 GELU/Softmax 来源："
                f"{stage2_fixed_label} (source={stage2_fixed_source})"
            )
            self.log(f"  Stage-2 GELU   : {np.asarray(stage2_fixed_gelu).tolist()}")
            self.log(f"  Stage-2 Softmax: {np.asarray(stage2_fixed_softmax).tolist()}")

        previous_log_file = getattr(self, "active_log_file", self.log_file)
        self._initialize_noise_log_file()
        self.active_log_file = self.noise_log_file
        try:
            if self.skip_noise_rl:
                self.log("\n[信息] 第二阶段噪声RL训练已跳过（--skip-noise-rl）。")
                if self.run_output_dir:
                    update_persistent_metadata_stage(
                        self.run_output_dir, "stage2_search", "skipped")
            else:
                noise_stage_result = self.run_noise_rl_stage(
                    fixed_gelu=stage2_fixed_gelu,
                    fixed_softmax=stage2_fixed_softmax,
                    fixed_label=stage2_fixed_label,
                    fixed_source=stage2_fixed_source,
                    resume_checkpoint_path=self._get_stage2_resume_checkpoint_path(),
                )
                if self.run_output_dir:
                    stage2_status = str(
                        noise_stage_result.get("status", "completed")
                    )
                    completed_stage2_episodes = int(
                        noise_stage_result.get(
                            "blb_v3_total_episodes",
                            self.stage2_rl_episodes,
                        )
                    )
                    update_persistent_metadata_stage(
                        self.run_output_dir, "stage2_search", stage2_status,
                        extra_fields={
                            "episodes": completed_stage2_episodes,
                            "configured_episode_limit": (
                                None
                                if self.stage2_rl_episodes == 0
                                else int(self.stage2_rl_episodes)
                            ),
                        },
                    )
                if str(noise_stage_result.get("status") or "") in (
                        "completed", "completed_infeasible"):
                    stage1_payload = getattr(
                        self, "stage1_best_config_payload", None
                    )
                    if not isinstance(stage1_payload, Mapping):
                        raise RuntimeError(
                            "completed Stage 2 is missing its Stage-1 JSON payload"
                        )
                    self._write_search_best_config_file(
                        stage1_config=stage1_payload["stage1"],
                        stage2_result=noise_stage_result,
                    )


            if self.skip_final_eval:
                self.log("\n[信息] 统一最终评估已跳过（--skip-final-eval）。")
                if self.run_output_dir:
                    update_persistent_metadata_stage(
                        self.run_output_dir, "final_eval", "skipped")
            else:
                if not self.final_eval_only:
                    raise RuntimeError(
                        "final evaluation is a separate JSON-driven command"
                    )
                from rfr.search.common.best_config import load_search_best_config

                search_config = load_search_best_config(
                    self.search_best_config_input_path,
                    require_final_eval_eligible=True,
                )
                expected_algorithm = (
                    "rl" if self.blb_v3_search_backend == "ppo"
                    else self.blb_v3_search_backend
                )
                expected_model = (
                    "bert-large" if int(self.total_layers) == 24 else "bert-base"
                )
                if (
                        search_config["algorithm"] != expected_algorithm
                        or search_config["model_type"] != expected_model
                        or search_config["dataset"] != self.dataset_key
                ):
                    raise ValueError(
                        "search-best JSON does not match the loaded evaluation profile"
                    )
                final_eval_result = self.run_selected_config_final_eval(
                    search_config=search_config,
                )
                if self.run_output_dir:
                    update_persistent_metadata_stage(
                        self.run_output_dir,
                        "final_eval",
                        "completed",
                    )
        finally:
            self.active_log_file = previous_log_file

        self.last_noise_stage_result = noise_stage_result
        self.last_final_eval_result = final_eval_result

        if final_eval_result is not None:
            opt_gelu = np.asarray(final_eval_result["opt_gelu"], dtype=int)
            opt_softmax = np.asarray(final_eval_result["opt_softmax"], dtype=int)
        elif best_config is not None:
            opt_gelu = np.asarray(best_config["gelu"], dtype=int)
            opt_softmax = np.asarray(best_config["softmax"], dtype=int)
        else:
            opt_gelu = stage2_fixed_gelu
            opt_softmax = stage2_fixed_softmax

        self.log("\n配置评估完成（Configuration evaluation finished）。")

        try:
            self.save_best_policies_snapshot()
        except Exception as _bp_err:
            self.log(f"  [best_policy][警告] 汇总失败：{_bp_err}")
        self.apply_configuration(opt_gelu, opt_softmax)
