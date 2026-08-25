"""Stage-1 PPO search and the shared Stage-2/final-evaluation handoff."""

from __future__ import annotations

from collections import deque
import time
import copy
import itertools
import math
import sys
import threading
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
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
    INPUT_NOISE_ALLOWED_SCALING_FACTORS,
    INPUT_NOISE_DEFAULT_SCALING_FACTOR,
    WEIGHT_NOISE_ALLOWED_SCALING_FACTORS,
    WEIGHT_NOISE_DEFAULT_SCALING_FACTOR,
    WFFN1_NOISE_ALLOWED_SCALING_FACTORS,
    WFFN1_NOISE_DEFAULT_SCALING_FACTOR,
    SOFTMAX_VALUE_NOISE_ALLOWED_SCALING_FACTORS,
    SOFTMAX_VALUE_NOISE_DEFAULT_SCALING_FACTOR,
)
from final_evaluation_module import UnifiedFinalEvaluationModule
from rfr.preparation.data.protocol import TRAIN_PROBE_SPLIT, validate_dataset
from rfr.preparation.data.mrpc_reproducibility import (
    MRPC_STAGE2_RL_ALIGNMENT_BATCH_SIZE,
    validate_mrpc_evaluation_setup,
)
from rfr.search.runtime.control import (
    PROGRESS_BOX_PPO_INTERVAL as NOISE_RL_PROGRESS_BOX_PPO_INTERVAL,
    STOP_FLAG_FILENAME as NOISE_STAGE_STOP_FLAG_FILENAME,
    consume_stop_flag as consume_stop_flag_file,
    format_eta_finish as _fmt_eta_finish,
    graceful_stop_requested as is_graceful_stop_requested,
    install_graceful_stop_handler,
    log_box as _log_rounded_box,
    reset_graceful_stop_state,
    uninstall_graceful_stop_handler,
    write_warning_report as _write_warning_report,
)
from rfr.common.report_format_utils import format_elapsed as _fmt_elapsed
from rfr.common.report_format_utils import progress_bar as _progress_bar
from rfr.search.runtime.elastic_gpu import (
    ElasticGPUFailure,
    is_recoverable_gpu_failure,
    raise_if_elastic_gpu_restart_requested,
)
from rfr.search.common.data_points import (
    RLDataPointWriter,
    make_unique_run_id,
    write_dataset_protocol,
)
import os
import random
import hashlib


def _build_ordinary_two_stage_result(
        *,
        backend: str,
        stage1_best_config: Mapping[str, Any],
        stage2_result: Mapping[str, Any],
        final_eval_result: Mapping[str, Any] | None,
        final_eval_status: str,
        final_eval_ineligible_reason: str | None,
        final_eval_error: str | None,
        ) -> dict[str, Any]:
    """Build the trusted-environment two-stage comparator summary."""
    from rfr.common.json_utils import to_jsonable

    normalized_backend = str(backend or "")
    stage1_binding = dict(
        stage1_best_config.get("selection_binding") or {}
    )
    consumed_binding = dict(
        stage2_result.get("stage1_consumed_binding") or {}
    )
    for binding in (stage1_binding, consumed_binding):
        result_path = binding.get("result_path")
        if result_path:
            binding["result_path"] = os.path.abspath(
                os.path.expanduser(os.fspath(result_path))
            )
    num_layers = int(stage1_binding.get("num_layers", 0) or 0)
    if (
            not normalized_backend
            or stage1_binding.get("backend") != normalized_backend
            or stage2_result.get("search_backend") != normalized_backend
            or num_layers <= 0
            or len(stage1_binding.get("action") or ()) != num_layers
            or len(stage1_binding.get("gelu_degrees") or ()) != num_layers
            or len(stage1_binding.get("softmax_degrees") or ()) != num_layers
            or any(
                int(value) != 6
                for value in stage1_binding.get("softmax_degrees") or ()
            )
    ):
        raise RuntimeError(
            "two-stage comparator has an invalid Stage-1 selection binding"
        )
    if consumed_binding != stage1_binding:
        raise RuntimeError(
            "Stage-2 result does not match the Stage-1 selection binding"
        )

    stage2_status = str(stage2_result.get("status") or "")
    strict_feasible = bool(stage2_result.get("strict_feasible", False))
    status_contract = {
        "completed": (True, "complete_strict_feasible"),
        "completed_infeasible": (False, "complete_least_violating"),
        "smoke_only_complete": (False, "smoke_only_complete"),
    }
    if stage2_status not in status_contract:
        raise RuntimeError(
            f"two-stage comparator has invalid Stage-2 status {stage2_status!r}"
        )
    expected_feasible, outer_status = status_contract[stage2_status]
    if strict_feasible is not expected_feasible:
        raise RuntimeError(
            "two-stage comparator Stage-2 status and strict verdict disagree"
        )
    if final_eval_result is not None and not strict_feasible:
        raise RuntimeError(
            "strict-infeasible selection cannot include final evaluation"
        )

    selection_diagnostics = dict(
        stage2_result.get("selection_diagnostics") or {}
    )
    artifact_paths = dict(
        selection_diagnostics.get("artifact_paths") or {}
    )
    action_group = dict(
        stage2_result.get("blb_v3_best_action_group") or {}
    )
    action_matrix = action_group.get("policy_actions")
    if stage2_status != "smoke_only_complete" and not isinstance(
            action_matrix, (list, tuple)
    ):
        raise RuntimeError(
            "two-stage comparator Stage-2 result has no layerwise action matrix"
        )

    return to_jsonable({
        "schema_version": "two_stage_search_result_v1",
        "backend": normalized_backend,
        "status": outer_status,
        "strict_feasible": strict_feasible,
        "stage1_bound_into_stage2": True,
        "smoke_only": stage2_status == "smoke_only_complete",
        "stage1": {
            "selection_binding": stage1_binding,
            "result_path": stage1_binding.get("result_path"),
            "gelu_degrees": list(stage1_binding["gelu_degrees"]),
            "softmax_degrees": list(stage1_binding["softmax_degrees"]),
            "evaluation": stage1_best_config.get("evaluation"),
            "search_accounting": stage1_best_config.get(
                "search_accounting"
            ),
        },
        "stage2": {
            "status": stage2_status,
            "strict_feasible": strict_feasible,
            "consumed_stage1_binding": consumed_binding,
            "manifest_path": artifact_paths.get("manifest"),
            "selected_configuration_path": artifact_paths.get(
                "final_selected_configuration"
            ),
            "strict_validation_path": artifact_paths.get(
                "strict_validation"
            ),
            "action_vec": stage2_result.get("blb_v3_best_action_vec"),
            "action_group": action_group,
            "layerwise_configuration": stage2_result.get(
                "blb_v3_layerwise_best_configuration"
            ),
            "final_config_fingerprint": stage2_result.get(
                "final_config_fingerprint"
            ),
            "search_accounting": stage2_result.get("search_accounting"),
            "selection_diagnostics": selection_diagnostics,
        },
        "final_eval": {
            "status": str(final_eval_status),
            "executed": final_eval_result is not None,
            "result_path": (
                final_eval_result.get("summary_path")
                if isinstance(final_eval_result, Mapping) else None
            ),
            "ineligible_reason": final_eval_ineligible_reason,
            "error": final_eval_error,
            "result": final_eval_result,
        },
    }, stringify_unknown=True)


GELU_MAP = {0: 4, 1: 2, 2: 1, 3: 0}
GELU_COST = {4: 3.0, 2: 2.5, 1: 1.0, 0: -1.0}
STAGE1_GELU_ACTION_MASK = np.array([True, True, True, False], dtype=bool)
SOFTMAX_COST = {6: 3.0, 5: 2.5, 4: 2.0, 3: 1.5, 2: 1.0}


FIXED_SOFTMAX_DEGREE = 6
STAGE1_ORIGINAL_FUNCTION_DEGREE = -1


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
        if gelu_degrees[idx] == STAGE1_ORIGINAL_FUNCTION_DEGREE
    ]
    if original_gelu_layers:
        handler.restore_layer_gelu(original_gelu_layers, handler_layer_name)

    original_softmax_layers = [
        idx for idx in softmax_indices
        if softmax_degrees[idx] == STAGE1_ORIGINAL_FUNCTION_DEGREE
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


_NOISE_COST_SCALE = 0.025
INPUT_NOISE_COST_MAP = {
    scaling_factor: scaling_factor * _NOISE_COST_SCALE
    for scaling_factor in INPUT_NOISE_ALLOWED_SCALING_FACTORS
}
WEIGHT_NOISE_COST_MAP = {
    scaling_factor: scaling_factor * _NOISE_COST_SCALE
    for scaling_factor in WEIGHT_NOISE_ALLOWED_SCALING_FACTORS
}
WFFN1_NOISE_COST_MAP = {
    scaling_factor: scaling_factor * _NOISE_COST_SCALE
    for scaling_factor in WFFN1_NOISE_ALLOWED_SCALING_FACTORS
}


NOISE_STAGE_STEP_INFO_FILE = "noise_ppo_step_info.txt"
NOISE_STAGE_TRAINING_CURVE_PATH = "noise_ppo_training_curve.png"
NOISE_STAGE_ENTROPY_CURVE_PATH = "noise_ppo_entropy_curve.png"
DEFAULT_STAGE1_SEARCH_LOG_FILE = "pruning_search_log.txt"

SEARCH_LOG_HEADER = "=== PPO强化学习优化日志已启动（PPO RL Optimization Log Started） ==="
DEFAULT_STAGE1_STEP_INFO_FILE = "ppo_step_info.txt"
DEFAULT_STAGE1_TRAINING_CURVE_FILE = "ppo_training_curve.png"
DEFAULT_STAGE1_ENTROPY_CURVE_FILE = "ppo_entropy_curve.png"
DEFAULT_FINAL_EVAL_DIR = os.path.join("Parting Chapter", "final_eval")
DEFAULT_NOISE_PROGRESS_DIR = os.path.join("Parting Chapter", "noise_rl_progress")


def ensure_parent_dir(path: str) -> None:
    """Create the parent directory of ``path`` if it doesn't exist (mkdir -p
    semantics). Idempotent. Used before atomic-writing JSON / pickle / etc."""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def update_persistent_metadata_stage(
        run_output_dir: str,
        stage_key: str,
        status: str,
        extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    """更新持久化目录中 metadata.json 的阶段完成状态。

    stage_key: 'stage1_search', 'stage2_search', 'final_eval'
    status:    'converged', 'budget_exhausted', 'completed', 'skipped',
               'in_progress', 'not_started'
    extra_fields: dict of additional fields to merge (optional)
    """
    import json as _json
    import datetime as _dt
    meta_path = os.path.join(run_output_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        return
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = _json.load(f)
    except Exception:
        return
    stages = meta.setdefault("stage_status", {})
    stages[stage_key] = status
    if extra_fields:
        stage_detail = meta.setdefault("stage_detail", {})
        stage_detail.setdefault(stage_key, {}).update(extra_fields)
    meta["last_updated_at"] = _dt.datetime.now().isoformat()
    tmp_path = meta_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            _json.dump(meta, f, indent=2, ensure_ascii=False)

        os.replace(tmp_path, meta_path)
    except Exception:
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def resolve_run_output_layout(run_output_dir, flattened=False):
    s1_log = s2_log = DEFAULT_STAGE1_SEARCH_LOG_FILE
    run_output_dir = str(run_output_dir or "").strip()
    if not run_output_dir:
        return {
            "run_output_dir": "",
            "log_file": s1_log,
            "noise_log_file": s2_log,
            "stage1_step_info_file": DEFAULT_STAGE1_STEP_INFO_FILE,
            "stage1_training_curve_path": DEFAULT_STAGE1_TRAINING_CURVE_FILE,
            "stage1_entropy_curve_path": DEFAULT_STAGE1_ENTROPY_CURVE_FILE,
            "final_eval_dir": DEFAULT_FINAL_EVAL_DIR,
            "noise_step_info_file": NOISE_STAGE_STEP_INFO_FILE,
            "noise_training_curve_path": NOISE_STAGE_TRAINING_CURVE_PATH,
            "noise_entropy_curve_path": NOISE_STAGE_ENTROPY_CURVE_PATH,
            "noise_progress_dir": DEFAULT_NOISE_PROGRESS_DIR,
        }

    run_output_dir = os.path.normpath(run_output_dir)
    if flattened:


        stage1_dir = run_output_dir
        stage2_noise_dir = run_output_dir
    else:
        stage1_dir = os.path.join(run_output_dir, "stage1")
        stage2_noise_dir = os.path.join(run_output_dir, "stage2_noise")
    stage2_noise_progress_dir = os.path.join(stage2_noise_dir, "progress")
    final_eval_dir = os.path.join(run_output_dir, "final_eval")

    layout = {
        "run_output_dir": run_output_dir,
        "log_file": os.path.join(stage1_dir, s1_log),
        "noise_log_file": os.path.join(stage2_noise_dir, s2_log),
        "stage1_step_info_file": os.path.join(stage1_dir, DEFAULT_STAGE1_STEP_INFO_FILE),
        "stage1_training_curve_path": os.path.join(
            stage1_dir, DEFAULT_STAGE1_TRAINING_CURVE_FILE
        ),
        "stage1_entropy_curve_path": os.path.join(
            stage1_dir, DEFAULT_STAGE1_ENTROPY_CURVE_FILE
        ),
        "final_eval_dir": final_eval_dir,
        "noise_step_info_file": os.path.join(stage2_noise_dir, NOISE_STAGE_STEP_INFO_FILE),
        "noise_training_curve_path": os.path.join(
            stage2_noise_dir, NOISE_STAGE_TRAINING_CURVE_PATH
        ),
        "noise_entropy_curve_path": os.path.join(
            stage2_noise_dir, NOISE_STAGE_ENTROPY_CURVE_PATH
        ),
        "noise_progress_dir": stage2_noise_progress_dir,
    }

    return layout


PPO_GAMMA = 0.99
PPO_LAMBDA = 0.95
PPO_EPS_CLIP = 0.2
PPO_K_EPOCHS = 4
PPO_VALUE_COEF = 0.5

PPO_MAX_EPISODES = 51000


PPO_UPDATE_INTERVAL = 120
PPO_BATCH_SIZE = 12 * PPO_UPDATE_INTERVAL
STEP_INFO_CHUNK_SIZE = 3 * PPO_UPDATE_INTERVAL


def set_ppo_update_interval(n):
    """Set the rollout interval before constructing the evaluator."""
    global PPO_UPDATE_INTERVAL, PPO_BATCH_SIZE, STEP_INFO_CHUNK_SIZE
    n = int(n)
    if n <= 0:
        raise ValueError(f"ppo_update_interval 必须是正整数, 得到 {n}")
    PPO_UPDATE_INTERVAL = n
    PPO_BATCH_SIZE = 12 * n
    STEP_INFO_CHUNK_SIZE = 3 * n


REWARD_DROP_WARNING_THRESHOLD = 0.2


PPO_LR_INITIAL = 5e-5
PPO_ENTROPY_INITIAL = 0.01


REWARD_THRESHOLD = 0.005
REWARD_COST_WEIGHT = 20.0
REWARD_DENSE_SCALE = 0.1

REWARD_CLIP_MIN = -5.0
REWARD_CLIP_MAX = 5.0


REWARD_NORMALIZATION_SCALE = 20.0


VALUE_CLIP_RANGE = 0.2


RUNNING_REWARD_HISTORY_SIZE = 100
RUNNING_REWARD_MIN_SAMPLES = 10
RUNNING_REWARD_EPSILON = 1e-8


HISTORY_MASK_VALUE = 0.0

POLICY_CONTINUOUS_DIM = 6
POLICY_POSITION_DIM = 16
POLICY_ACTION_DIM = 8
POLICY_PROJECTION_DIM = 32
SOS_TOKEN_GELU = 4


GTRXL_D_MODEL = 64
GTRXL_N_HEADS = 4
GTRXL_N_LAYERS = 3
GTRXL_D_FF = 128
GTRXL_DROPOUT = 0.1
GTRXL_WARMUP_STEPS = 500
GTRXL_ENTROPY_LOWER_BOUND = 0.012
GTRXL_ENTROPY_RECOVERY_MULTIPLIER = 25.0
GTRXL_KL_TARGET = 0.02
GTRXL_MINI_BATCH_EPISODES = 8

def resolve_ppo_learning_rate(raw_value, default_lr=PPO_LR_INITIAL):
    """Resolve a positive PPO learning rate."""
    if raw_value is None or raw_value == "":
        return float(default_lr), "default"

    try:
        lr = float(raw_value)
    except (TypeError, ValueError):
        return float(default_lr), "default"

    if not 0.0 < lr < 1.0:
        return float(default_lr), "default"
    return float(lr), "direct"


DATASET_METRICS_CONFIG = {
    dataset: {
        'type': 'classification',
        'metrics': ['accuracy', 'weighted_f1'],
        'metric_names': ['Acc.', 'F1'],
        'metric_full_names': ['Accuracy', 'Weighted F1'],
    }
    for dataset in ('mrpc', 'rte', 'sst2')
}


LOG_BARRIER_VIOLATION_SCALE = 10.0
LOG_BARRIER_VIOLATION_STEEPNESS = 20.0
LOG_BARRIER_SATISFACTION_SCALE = 0.5


PPO_ENTROPY_START = 0.05
PPO_ENTROPY_END = 0.001


from rfr.search.common.local_optimum import detect_rl_local_optimum  # noqa: E402,F401


class RunningMeanStd:
    """Track return moments with Welford's online update."""
    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x):
        if isinstance(x, torch.Tensor):
            x_detached = x.detach()
            if not torch.is_floating_point(x_detached):
                x_detached = x_detached.float()
            batch_mean = float(x_detached.mean().item())
            batch_var = float(x_detached.var(unbiased=False).item())
            batch_count = int(x_detached.numel())
            self._update_from_moments(batch_mean, batch_var, batch_count)
            return
        x = np.asarray(x).flatten()

        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        """Welford's parallel algorithm for combining statistics"""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count


        self.mean = self.mean + delta * batch_count / total_count


        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        self.var = M2 / total_count

        self.count = total_count

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)

    def normalize(self, x):
        """归一化数据"""
        if isinstance(x, torch.Tensor):
            return (x - self.mean) / (self.std + 1e-8)
        return (x - self.mean) / (self.std + 1e-8)

    def denormalize(self, x):
        """反归一化数据"""
        return x * self.std + self.mean


class GRUGate(nn.Module):
    """Gate a GTrXL residual branch with a trainable carry bias."""
    def __init__(self, d_model):
        super().__init__()
        self.Wr = nn.Linear(d_model, d_model, bias=False)
        self.Ur = nn.Linear(d_model, d_model, bias=False)
        self.Wz = nn.Linear(d_model, d_model, bias=False)
        self.Uz = nn.Linear(d_model, d_model, bias=False)
        self.Wg = nn.Linear(d_model, d_model, bias=False)
        self.Ug = nn.Linear(d_model, d_model, bias=False)
        self.bg = nn.Parameter(torch.ones(d_model) * 2.0)

    def forward(self, x, y):
        r = torch.sigmoid(self.Wr(y) + self.Ur(x))
        z = torch.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h_hat = torch.tanh(self.Wg(y) + self.Ug(r * x))
        return (1 - z) * x + z * h_hat


class GTrXLBlock(nn.Module):
    """Pre-norm causal attention and feed-forward layers with GRU gates."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.gate1 = GRUGate(d_model)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.gate2 = GRUGate(d_model)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        norm_x = self.ln1(x)
        attn_out, _ = self.attn(
            norm_x, norm_x, norm_x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        x = self.gate1(x, attn_out)

        norm_x2 = self.ln2(x)
        ff_out = self.ff(norm_x2)
        x = self.gate2(x, ff_out)
        return x


class GTrXLStrategyNetwork(nn.Module):
    """Shared causal GTrXL actor-critic used by the Stage-1 policy."""
    def __init__(self, num_layers=12, d_model=GTRXL_D_MODEL,
                 n_heads=GTRXL_N_HEADS, n_gtrxl_layers=GTRXL_N_LAYERS,
                 d_ff=GTRXL_D_FF, dropout=GTRXL_DROPOUT):
        super(GTrXLStrategyNetwork, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model


        self.embed_layer_idx = nn.Embedding(num_layers, POLICY_POSITION_DIM)
        self.embed_prev_g = nn.Embedding(SOS_TOKEN_GELU + 1, POLICY_ACTION_DIM)


        self.fc_continuous = nn.Sequential(
            nn.Linear(POLICY_CONTINUOUS_DIM, POLICY_PROJECTION_DIM),
            nn.LayerNorm(POLICY_PROJECTION_DIM),
            nn.SiLU()
        )


        token_input_dim = (
            POLICY_POSITION_DIM + POLICY_ACTION_DIM + POLICY_PROJECTION_DIM
        )
        self.input_proj = nn.Identity() if token_input_dim == d_model else nn.Linear(token_input_dim, d_model)


        self.gtrxl_blocks = nn.ModuleList([
            GTrXLBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_gtrxl_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)


        self.actor_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh()
        )
        self.head_g = nn.Linear(64, 4)


        self.critic_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self._initialize_weights()
        self._causal_mask_cache = {}

    def _initialize_weights(self):
        for module in [self.actor_head, self.critic_head, self.fc_continuous]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)

        nn.init.orthogonal_(self.head_g.weight, gain=0.01)
        nn.init.constant_(self.head_g.bias, 0.0)

        if isinstance(self.input_proj, nn.Linear):
            nn.init.orthogonal_(self.input_proj.weight, gain=1.0)
            if self.input_proj.bias is not None:
                nn.init.constant_(self.input_proj.bias, 0.0)

        for block in self.gtrxl_blocks:
            for p in block.attn.in_proj_weight.chunk(3):
                nn.init.orthogonal_(p)
            nn.init.orthogonal_(block.attn.out_proj.weight)
            for layer in block.ff:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)

    def _get_causal_mask(self, seq_len, device):
        """Return a cached upper-triangular causal mask."""
        if seq_len not in self._causal_mask_cache or self._causal_mask_cache[seq_len].device != device:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
            self._causal_mask_cache[seq_len] = mask
        return self._causal_mask_cache[seq_len]

    def _build_tokens(self, cont_features, layer_indices, prev_g_actions):
        emb_l = self.embed_layer_idx(layer_indices)
        emb_pg = self.embed_prev_g(prev_g_actions)
        feat_c = self.fc_continuous(cont_features)

        token_input = torch.cat([emb_l, emb_pg, feat_c], dim=-1)
        return self.input_proj(token_input)

    def forward(self, cont_features, layer_indices, prev_g_actions,
                gelu_mask=None, key_padding_mask=None):
        tokens = self._build_tokens(cont_features, layer_indices, prev_g_actions)
        seq_len = tokens.size(1)
        causal_mask = self._get_causal_mask(seq_len, tokens.device)

        x = tokens
        for block in self.gtrxl_blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
        x = self.ln_final(x)

        actor_feat = self.actor_head(x)
        logits_g = self.head_g(actor_feat)

        if gelu_mask is not None:
            logits_g = logits_g.masked_fill(~gelu_mask, float('-inf'))

        values = self.critic_head(x).squeeze(-1)

        return logits_g, values

    def get_action_and_logprob(self, cont_features, layer_indices, prev_g_actions,
                                return_probs=False, gelu_mask=None, key_padding_mask=None):
        logits_g, values = self.forward(
            cont_features, layer_indices, prev_g_actions,
            gelu_mask=gelu_mask, key_padding_mask=key_padding_mask
        )

        lg = logits_g[:, -1, :]
        val = values[:, -1]

        lg = lg.squeeze(0)
        val = val.squeeze(0)

        dist_g = Categorical(logits=lg)
        action_g = dist_g.sample()
        logprob = dist_g.log_prob(action_g)

        if return_probs:
            g_probs = torch.softmax(lg, dim=-1)
            return action_g, logprob, val, g_probs

        return action_g, logprob, val

    def evaluate_actions(self, cont_features, layer_indices, prev_g_actions,
                         actions_g, gelu_mask=None):
        logits_g, values = self.forward(
            cont_features, layer_indices, prev_g_actions,
            gelu_mask=gelu_mask
        )

        dist_g = Categorical(logits=logits_g)

        logprobs = dist_g.log_prob(actions_g)
        entropy = dist_g.entropy()

        return logprobs, entropy, values


def _pack_recurrent_rollout_arrays(episodes):
    if not episodes:
        raise RuntimeError("RecurrentRolloutBuffer is empty")
    layer_indices_np = np.asarray([ep['layer_indices'] for ep in episodes], dtype=np.int64)
    prev_g_actions_np = np.asarray([ep['prev_g_actions'] for ep in episodes], dtype=np.int64)
    actions_g_np = np.asarray([ep['actions_g'] for ep in episodes], dtype=np.int64)
    rewards_np = np.asarray([ep['rewards'] for ep in episodes], dtype=np.float32)
    dones_np = np.asarray([ep['dones'] for ep in episodes], dtype=np.float32)
    has_masks = all(len(ep.get('gelu_masks', [])) > 0 for ep in episodes)
    gelu_masks_np = (
        np.asarray([ep['gelu_masks'] for ep in episodes], dtype=bool)
        if has_masks
        else None
    )
    return (
        layer_indices_np,
        prev_g_actions_np,
        actions_g_np,
        rewards_np,
        dones_np,
        gelu_masks_np,
    )


def _rollout_tensor_to_numpy(value, dtype):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _stage1_scalar_tensors_to_float_list(values):
    if not values:
        return []
    stacked = torch.stack([v.detach().reshape(()) for v in values], dim=0)
    return [float(x) for x in stacked.detach().cpu().numpy().reshape(-1)]


def _stage1_prob_tensors_to_nested_lists(values):
    if not values:
        return []
    stacked = torch.stack([v.detach() for v in values], dim=0)
    return stacked.detach().cpu().numpy().tolist()


def _stage1_scalar_rows_to_numpy(rows):
    return np.asarray([
        [
            float(value.detach().cpu().item()) if hasattr(value, "detach") else float(value)
            for value in row
        ]
        for row in rows
    ], dtype=np.float32)


def _stage1_scalar_episode_values_to_tensor(episodes, field, device):
    rows = [ep[field] for ep in episodes]
    device = torch.device(device)
    if not rows:
        return torch.empty((0, 0), dtype=torch.float32, device=device)
    expected_len = len(rows[0])
    if any(len(row) != expected_len for row in rows):
        return torch.as_tensor(
            _stage1_scalar_rows_to_numpy(rows),
            dtype=torch.float32,
            device=device,
        )

    flat_values = [value for row in rows for value in row]
    if not flat_values:
        return torch.empty((len(rows), 0), dtype=torch.float32, device=device)
    if all(hasattr(value, "detach") for value in flat_values):
        stacked = torch.stack(
            [value.detach().reshape(()) for value in flat_values],
            dim=0,
        ).reshape(len(episodes), expected_len)
        return stacked.to(device=device, dtype=torch.float32)
    if any(hasattr(value, "detach") for value in flat_values):
        return torch.as_tensor(
            _stage1_scalar_rows_to_numpy(rows),
            dtype=torch.float32,
            device=device,
        )
    return torch.as_tensor(rows, dtype=torch.float32, device=device)


_stage1_gelu_mask_template_cache: Dict[Tuple[str, Tuple[bool, ...]], Tuple[np.ndarray, torch.Tensor]] = {}


def _get_stage1_gelu_mask_templates(device):
    """Static Stage-1 GELU action mask as reusable numpy + device tensors."""
    device = torch.device(device)
    mask_key = tuple(bool(x) for x in STAGE1_GELU_ACTION_MASK.tolist())
    key = (str(device), mask_key)
    cached = _stage1_gelu_mask_template_cache.get(key)
    if cached is not None:
        return cached

    mask_np = np.asarray(STAGE1_GELU_ACTION_MASK, dtype=bool).copy()
    mask_t = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
    mask_np.setflags(write=False)
    cached = (mask_np, mask_t)
    _stage1_gelu_mask_template_cache[key] = cached
    return cached


def _pack_recurrent_rollout_tensor_arrays(episodes, device):
    if not episodes:
        raise RuntimeError("RecurrentRolloutBuffer is empty")
    cont_features_np = np.asarray([
        [_rollout_tensor_to_numpy(t, np.float32) for t in ep['cont_features']]
        for ep in episodes
    ], dtype=np.float32)
    logprobs = _stage1_scalar_episode_values_to_tensor(episodes, 'logprobs', device)
    values = _stage1_scalar_episode_values_to_tensor(episodes, 'values', device)
    return cont_features_np, logprobs, values


class RecurrentRolloutBuffer:
    """Store complete causal episodes for batched GTrXL PPO updates."""
    def __init__(self):
        self.episodes = []
        self._current = None

    def start_episode(self):
        """开始记录新的Episode"""
        self._current = {
            'cont_features': [],
            'layer_indices': [],
            'prev_g_actions': [],
            'actions_g': [],
            'logprobs': [],
            'rewards': [],
            'values': [],
            'dones': [],
            'gelu_masks': [],
        }

    def add_step(self, cont_feat, layer_idx, prev_g,
                 action_g, logprob, reward, value, done, gelu_mask=None):
        """添加一步数据到当前Episode（gelu-only）"""
        self._current['cont_features'].append(cont_feat)
        self._current['layer_indices'].append(layer_idx)
        self._current['prev_g_actions'].append(prev_g)
        self._current['actions_g'].append(action_g)
        self._current['logprobs'].append(logprob)
        self._current['rewards'].append(reward)
        self._current['values'].append(value)
        self._current['dones'].append(done)
        if gelu_mask is not None:
            self._current['gelu_masks'].append(gelu_mask)

    def end_episode(self):
        """结束当前Episode，加入存储"""
        self.episodes.append(self._current)
        self._current = None

    def clear(self):
        """清空所有存储"""
        self.episodes.clear()

    @property
    def num_episodes(self):
        return len(self.episodes)

    def get_batch(self, device):
        """Stack buffered episodes into device-ready tensors."""
        cont_features_np, logprobs, values = _pack_recurrent_rollout_tensor_arrays(
            self.episodes,
            device,
        )
        cont_features = torch.from_numpy(cont_features_np).to(device)

        (
            layer_indices_np,
            prev_g_actions_np,
            actions_g_np,
            rewards_np,
            dones_np,
            gelu_masks_np,
        ) = _pack_recurrent_rollout_arrays(self.episodes)

        layer_indices = torch.from_numpy(layer_indices_np).to(device)

        prev_g_actions = torch.from_numpy(prev_g_actions_np).to(device)

        actions_g = torch.from_numpy(actions_g_np).to(device)

        rewards = torch.from_numpy(rewards_np).to(device)

        dones = torch.from_numpy(dones_np).to(device)


        if gelu_masks_np is not None:
            gelu_masks = torch.from_numpy(gelu_masks_np).to(device)
        else:
            gelu_masks = None

        return (cont_features, layer_indices, prev_g_actions,
                actions_g, logprobs, rewards, values, dones, gelu_masks)


class TransformerOptEnv:
    """Stage-1 GELU search environment with fixed Softmax degree 6."""
    def __init__(self, total_layers, baseline_cost, baseline_metrics, evaluator,
                 constraint_limits=None, prev_metrics=None, num_metrics=2):
        self.total_layers = total_layers
        self.baseline_cost = baseline_cost
        self.baseline_loss, self.baseline_p, self.baseline_s = baseline_metrics
        self.evaluator = evaluator
        self.num_metrics = num_metrics


        if constraint_limits is None:

            self.constraint_limits = {
                'loss': self.baseline_loss * (1 + REWARD_THRESHOLD),
                'metric1': self.baseline_p * (1 - REWARD_THRESHOLD),
                'metric2': self.baseline_s * (1 - REWARD_THRESHOLD)
            }
        else:
            self.constraint_limits = constraint_limits


        if prev_metrics is None:
            self.prev_episode_metrics = {
                'loss': self.baseline_loss,
                'metric1': self.baseline_p,
                'metric2': self.baseline_s,
                'cost': self.baseline_cost
            }
        else:
            self.prev_episode_metrics = prev_metrics


        self.mid_gelu_cost = GELU_COST[2]
        self.mid_softmax_cost = SOFTMAX_COST[4]
        self.expected_cost_per_layer = self.mid_gelu_cost + self.mid_softmax_cost


        self.max_cost_per_layer = GELU_COST[4] + SOFTMAX_COST[6]


        self.gelu_degree_to_norm = {4: 0.0, 2: 0.5, 1: 1.0, 0: 1.25}

        self.current_episode_metrics = None

        self.reset()

    def reset(self):
        """重置环境"""
        self.current_layer = 0
        self.accumulated_cost = 0.0
        self.gelu_config = []
        self.softmax_config = []
        self.prev_gelu_degree = 4


        self.accumulated_dense_reward = 0.0


        self.gelu_history = np.full(self.total_layers, HISTORY_MASK_VALUE, dtype=np.float32)


        return self._get_state()

    def get_gelu_action_mask(self, layer_idx=None):
        """
        返回指定层的 GELU 动作掩码 (4-dim bool)。
        True = 该动作可选, False = 被禁止。
        动作索引: 0=degree4, 1=degree2, 2=degree1, 3=degree0(ReLU)。
        Stage-1 RL 当前禁用 degree 0；idx 3 保留给历史配置与手工 eval。

        如果 layer_idx 为 None，使用 self.current_layer。
        """
        del layer_idx
        return STAGE1_GELU_ACTION_MASK.copy()

    def _get_state(self):
        """
        构造31维状态向量（gelu-only：softmax 通道已移除）。

        Softmax 每层固定为 degree 6，不占动作或状态通道。活动的
        GTrXL 策略只消费 get_policy_cont_features() 暴露的 6 个连续特征，所以这个扁平
        向量的布局不再承载任何 softmax 通道。

        - 12维: 位置编码 (One-Hot)
        - 1维: 成本偏差 (Cost Deviation)
        - 1维: 上一步GELU动作编码
        - 1维: 累积复杂度债务 (Complexity Debt)
        - 1维: 进度指示 (Progress Indicator)
        - 12维: GELU动作历史（归一化，未访问层为0）
        - 3维: 预算感知 (loss/metric1/metric2 剩余预算)
        """


        position = np.zeros(self.total_layers)
        if self.current_layer < self.total_layers:
            position[self.current_layer] = 1.0


        expected_cost_so_far = self.current_layer * self.expected_cost_per_layer
        if expected_cost_so_far > 0:
            cost_deviation = (self.accumulated_cost - expected_cost_so_far) / expected_cost_so_far
        else:
            cost_deviation = 0.0

        cost_deviation = np.clip(cost_deviation, -1.0, 1.0)


        gelu_norm = self.gelu_degree_to_norm.get(self.prev_gelu_degree, 0.0)


        baseline_cost_so_far = self.current_layer * self.max_cost_per_layer
        if baseline_cost_so_far > 0:
            complexity_debt = (baseline_cost_so_far - self.accumulated_cost) / baseline_cost_so_far
        else:
            complexity_debt = 0.0

        complexity_debt = np.clip(complexity_debt, 0.0, 1.0)


        progress = self.current_layer / self.total_layers


        prev_loss = self.prev_episode_metrics['loss']
        prev_m1 = self.prev_episode_metrics['metric1']
        prev_m2 = self.prev_episode_metrics['metric2']


        loss_budget = 1.0 - prev_loss / (self.constraint_limits['loss'] + 1e-8)

        m1_budget = prev_m1 / (self.constraint_limits['metric1'] + 1e-8) - 1.0

        if self.num_metrics == 1:
            m2_budget = 0.0
        else:

            m2_budget = prev_m2 / (self.constraint_limits['metric2'] + 1e-8) - 1.0


        loss_budget = np.clip(loss_budget, -1.0, 1.0)
        m1_budget = np.clip(m1_budget, -1.0, 1.0)
        m2_budget = np.clip(m2_budget, -1.0, 1.0)


        self._policy_cont_features = np.array(
            [cost_deviation, complexity_debt, progress,
             loss_budget, m1_budget, m2_budget],
            dtype=np.float32,
        )
        state = np.concatenate([
            position,
            [cost_deviation],
            [gelu_norm],
            [complexity_debt],
            [progress],
            self.gelu_history,
            [loss_budget, m1_budget, m2_budget]
        ])
        return state.astype(np.float32)

    def get_policy_cont_features(self):
        """Return the 6-dim continuous feature vector the active GTrXL policy
        consumes: [cost_deviation, complexity_debt, progress, loss_budget,
        m1_budget, m2_budget]. Refreshed on every ``_get_state()`` call, so it is
        valid for whatever state ``reset()``/``step()`` most recently returned."""
        return self._policy_cont_features.copy()

    def _compute_dense_step_reward(self, gelu_degree):
        """
        策略一（3.1）：计算稠密化中间奖励（gelu-only）。

        Softmax 固定为 degree 6（成本为常数），因此每层成本奖励完全由 GELU degree
        驱动。Dense reward 仍对每层成本节约单调（无 expected-cost-track 偏置）。
        """
        step_cost = GELU_COST[gelu_degree] + SOFTMAX_COST[FIXED_SOFTMAX_DEGREE]


        cost_saving = (self.max_cost_per_layer - step_cost) / self.max_cost_per_layer
        cost_reward = REWARD_DENSE_SCALE * cost_saving
        return cost_reward

    def step(self, gelu_action_idx):
        """执行动作（gelu-only），返回(next_state, reward, done, info)。

        softmax 不再是动作：每层固定 degree 6（FIXED_SOFTMAX_DEGREE），其成本为常数。
        softmax_config 仍按层填入该固定 degree，供下游（stage1_evaluate / 报告）使用。
        """

        gelu_degree = GELU_MAP[gelu_action_idx]
        softmax_degree = FIXED_SOFTMAX_DEGREE


        self.gelu_config.append(gelu_degree)
        self.softmax_config.append(softmax_degree)


        self.accumulated_cost += GELU_COST[gelu_degree] + SOFTMAX_COST[softmax_degree]


        self.prev_gelu_degree = gelu_degree


        self.gelu_history[self.current_layer] = self.gelu_degree_to_norm[gelu_degree]


        dense_reward = self._compute_dense_step_reward(gelu_degree)
        self.accumulated_dense_reward += dense_reward


        info = {
            'layer_index': self.current_layer,
            'curr_gelu_degree': gelu_degree,
            'curr_softmax_degree': softmax_degree,
            'accumulated_cost': self.accumulated_cost,
            'gelu_config': self.gelu_config.copy(),
            'softmax_config': self.softmax_config.copy(),
            'dense_reward': dense_reward,
            'gelu_history': self.gelu_history.copy(),
        }

        self.current_layer += 1

        if self.current_layer < self.total_layers:

            return self._get_state(), dense_reward, False, info
        else:

            final_reward = self._compute_final_reward()
            info['final_reward'] = final_reward
            info['accumulated_dense_reward'] = self.accumulated_dense_reward

            total_reward = final_reward + dense_reward
            return self._get_state(), total_reward, True, info

    def _compute_final_reward(self):
        """Combine the terminal constraint barrier and cost saving."""

        gelu_arr = np.array(self.gelu_config)
        softmax_arr = np.array(self.softmax_config)


        loss, m1, m2, _ = self.evaluator.evaluate_model(gelu_arr, softmax_arr)


        self.current_episode_metrics = {
            'loss': loss,
            'metric1': m1,
            'metric2': m2,
            'cost': self.accumulated_cost
        }

        def log_barrier_reward(curr_value, limit_value, is_upper_bound=True):
            """
            对数障碍函数：当接近约束边界时梯度急剧增大

            Args:
                curr_value: 当前指标值
                limit_value: 约束阈值
                is_upper_bound: True表示约束为 curr < limit，False表示 curr > limit
            """
            if is_upper_bound:

                margin = limit_value - curr_value
            else:

                margin = curr_value - limit_value

            if margin < 0:

                return -LOG_BARRIER_VIOLATION_SCALE * np.exp(-margin * LOG_BARRIER_VIOLATION_STEEPNESS)
            else:

                return LOG_BARRIER_SATISFACTION_SCALE * np.log(margin + 1e-5)

        r_loss_barrier = log_barrier_reward(loss, self.constraint_limits['loss'], is_upper_bound=True)
        r_m1_barrier = log_barrier_reward(m1, self.constraint_limits['metric1'], is_upper_bound=False)
        r_m2_barrier = log_barrier_reward(m2, self.constraint_limits['metric2'], is_upper_bound=False)


        if self.num_metrics == 1:
            r_constraint = (r_loss_barrier + r_m1_barrier) / 2.0
        else:
            r_constraint = (r_loss_barrier + r_m1_barrier + r_m2_barrier) / 3.0


        cost_saving = (self.baseline_cost - self.accumulated_cost) / self.baseline_cost
        r_cost = cost_saving * REWARD_COST_WEIGHT


        r_accuracy = r_constraint


        raw_reward = r_accuracy + r_cost


        scaled_reward = raw_reward / REWARD_NORMALIZATION_SCALE


        clipped_reward = np.clip(scaled_reward, REWARD_CLIP_MIN, REWARD_CLIP_MAX)


        return clipped_reward


class LayerImportanceEvaluator(TrainerCallback):
    def __init__(self, model, train_data, test_data, data_collator, rl_lr=None, degree=None,
                 batch_size=16,
                 stage1_rl_episodes=PPO_MAX_EPISODES,
                 stage2_rl_episodes=0,
                 stage1_rl_episodes_specified=False,
                 stage2_rl_episodes_specified=False,
                 stage1_entropy_stop_threshold=None,
                 stage1_rl_lr=None,
                 stage2_rl_lr=None,
                 stage1_rl_devices="",
                 device='cuda', data_path='mrpc',
                 run_output_dir='',
                 final_eval_config_source='search',
                 final_eval_config_path='glue_final_configs_best_ppo.json',
                 manual_stage1_gelu=None,
                 manual_stage1_softmax=None,
                 manual_stage2_noise=None,
                 stage2_fixed_config_source='all4',
                 stage2_fixed_config_path='',
                 stage2_manual_gelu=None,
                 stage2_manual_softmax=None,
                 final_eval_random_seed=42,
                 final_eval_permutation_trials=10,
                 final_eval_cost_equivalent_trials=10,
                 final_eval_budget_equivalent_trials=10,
                 final_eval_stage1_budget_trials=10,
                 final_eval_stage2_budget_trials=10,
                 final_eval_repeat_n=5,
                 final_eval_preset='default',
                 final_eval_output_root='',
                 final_eval_run_name='',
                 final_eval_random_enabled=False,
                 final_eval_action_config='',
                 final_eval_action_ranges='',
                 final_eval_action_fixed='',
                 final_eval_cost_match_count=50,
                 final_eval_cost_match_max_attempts=5000,
                 skip_noise_rl=False,
                 skip_stage1_rl=False,
                 skip_final_eval=False,
                 final_eval_only=False,
                 resume_run_dir='',
                 decoupled_layout=False,
                 stage1_run_id='',
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
                  blb_v3_min_convergence_episodes=90000,
                  blb_v3_convergence_patience_updates=100,
                  blb_v3_search_backend="ppo",
                  blb_v3_search_evaluation_budget=0,
                  blb_v3_search_initial_design_size=64,
                  blb_v3_search_candidate_pool_size=2048,
                  blb_v3_search_population_size=64,
                  blb_v3_search_patience_generations=100,
                  blb_v3_search_mutation_max_coordinates=3,
                  blb_v3_search_rf_n_estimators=128,
                  blb_v3_search_rf_min_samples_leaf=2,
                  blb_v3_search_full_validation=True,
                  comparator_smoke=False,
                  comparator_stage1_only=False,
                  glue_data_protocol=None,
                  mrpc_reproducibility=None):
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
        self.mrpc_reproducibility = mrpc_reproducibility
        if self.mrpc_reproducibility is not None:
            if str(data_path).strip().lower() != "mrpc":
                raise ValueError(
                    "MRPC reproducibility fixture requires data_path='mrpc'"
                )
            validate_mrpc_evaluation_setup(
                model=self.model,
                tokenizer=getattr(data_collator, "tokenizer", None),
                collator=data_collator,
                full_validation=test_data,
                stability_probe=(
                    self.mrpc_reproducibility.stability_probe
                ),
                batch_size=self.batch_size,
            )


        from rfr.search.rl.stage1.eval_cache import Stage1EvalCache
        self._eval_cache = Stage1EvalCache()


        self._stage1_worker_eval_cache = Stage1EvalCache()
        self._stage1_parallel_timing_lock = threading.Lock()
        self._stage1_parallel_model_forward_seconds = 0.0
        self._stage1_parallel_model_forward_calls = 0

        self._eval_infra_ready = False


        self._last_applied_config = None
        if stage1_entropy_stop_threshold in (None, ""):
            self.stage1_entropy_stop_threshold = None
        else:
            self.stage1_entropy_stop_threshold = float(stage1_entropy_stop_threshold)
            if self.stage1_entropy_stop_threshold <= 0:
                raise ValueError(
                    "stage1_entropy_stop_threshold must be a positive float "
                    f"when set, got {stage1_entropy_stop_threshold!r}"
                )
        try:
            _stage1_episode_limit_raw = int(stage1_rl_episodes)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"stage1_rl_episodes must be an integer, got {stage1_rl_episodes!r}"
            ) from exc
        self.stage1_rl_unbounded_until_entropy = _stage1_episode_limit_raw <= 0
        if self.stage1_rl_unbounded_until_entropy:
            if self.stage1_entropy_stop_threshold is None:
                raise ValueError(
                    "stage1_rl_episodes <= 0 means unbounded Stage-1 training "
                    "and requires stage1_entropy_stop_threshold"
                )
            self.stage1_rl_episodes = _stage1_episode_limit_raw
            self.stage1_rl_episode_limit = None
        else:
            self.stage1_rl_episodes = self._coerce_positive_int(
                stage1_rl_episodes, 'stage1_rl_episodes'
            )
            self.stage1_rl_episode_limit = int(self.stage1_rl_episodes)
        self.stage2_rl_episodes = self._coerce_nonnegative_int(
            stage2_rl_episodes, 'stage2_rl_episodes'
        )
        self.stage2_rl_unbounded_until_convergence = self.stage2_rl_episodes == 0
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
        self.INPUT_NOISE_COST_MAP = INPUT_NOISE_COST_MAP.copy()
        self.WEIGHT_NOISE_COST_MAP = WEIGHT_NOISE_COST_MAP.copy()
        self.WFFN1_NOISE_COST_MAP = WFFN1_NOISE_COST_MAP.copy()


        self.current_gelu_degrees = np.full(self.total_layers, 4, dtype=int)
        self.current_softmax_degrees = np.full(self.total_layers, 6, dtype=int)
        self.current_input_noise_scaling_factors = np.full(
            self.total_layers,
            INPUT_NOISE_DEFAULT_SCALING_FACTOR,
            dtype=int,
        )
        self.input_noise_action_space = tuple(INPUT_NOISE_ALLOWED_SCALING_FACTORS)
        self.current_wq_noise_scaling_factors = np.full(
            self.total_layers,
            WEIGHT_NOISE_DEFAULT_SCALING_FACTOR,
            dtype=int,
        )
        self.current_wk_noise_scaling_factors = np.full(
            self.total_layers,
            WEIGHT_NOISE_DEFAULT_SCALING_FACTOR,
            dtype=int,
        )
        self.current_wv_noise_scaling_factors = np.full(
            self.total_layers,
            WEIGHT_NOISE_DEFAULT_SCALING_FACTOR,
            dtype=int,
        )
        self.current_wo_noise_scaling_factors = np.full(
            self.total_layers,
            WEIGHT_NOISE_DEFAULT_SCALING_FACTOR,
            dtype=int,
        )
        self.current_wffn1_noise_scaling_factors = np.full(
            self.total_layers,
            WFFN1_NOISE_DEFAULT_SCALING_FACTOR,
            dtype=int,
        )
        self.current_wffn2_noise_scaling_factors = np.full(
            self.total_layers,
            WEIGHT_NOISE_DEFAULT_SCALING_FACTOR,
            dtype=int,
        )
        self.current_softmax_value_softmax_noise_scaling_factors = np.full(
            self.total_layers,
            SOFTMAX_VALUE_NOISE_DEFAULT_SCALING_FACTOR,
            dtype=int,
        )
        self.current_softmax_value_v_noise_scaling_factors = np.full(
            self.total_layers,
            SOFTMAX_VALUE_NOISE_DEFAULT_SCALING_FACTOR,
            dtype=int,
        )
        self.wq_noise_action_space = tuple(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
        self.wk_noise_action_space = tuple(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
        self.wv_noise_action_space = tuple(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
        self.wo_noise_action_space = tuple(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
        self.wffn1_noise_action_space = tuple(WFFN1_NOISE_ALLOWED_SCALING_FACTORS)
        self.wffn2_noise_action_space = tuple(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
        self.softmax_value_noise_action_space = tuple(
            SOFTMAX_VALUE_NOISE_ALLOWED_SCALING_FACTORS
        )
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

        self.decoupled_layout = bool(decoupled_layout)
        self.stage1_run_id = str(stage1_run_id or "").strip()
        output_layout = resolve_run_output_layout(
            run_output_dir,
            flattened=self.decoupled_layout,
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
            _stage1_episode_desc = (
                f"unbounded until entropy < {self.stage1_entropy_stop_threshold:.6f}"
                if self.stage1_rl_unbounded_until_entropy
                else str(self.stage1_rl_episodes)
            )
            f.write(
                f"[信息] Stage-1 PPO学习率（LR）从 stage1_rl_lr={self.stage1_rl_lr_raw!r} 解析为 -> "
                f"{self.stage1_ppo_lr_initial:.6g} ({self.stage1_ppo_lr_mode}) | "
                f"Stage-2 PPO学习率（LR）从 stage2_rl_lr={self.stage2_rl_lr_raw!r} 解析为 -> "
                f"{self.stage2_ppo_lr_initial:.6g} ({self.stage2_ppo_lr_mode})\n"
            )
            f.write(
                f"[信息] 第一阶段RL回合数（Stage-1 RL episodes）: {_stage1_episode_desc} | "
                f"第二阶段RL回合数（Stage-2 RL episodes）: {self.stage2_rl_episodes}\n"
            )
            if self.stage1_entropy_stop_threshold is not None:
                f.write(
                    "[信息] Stage-1 entropy convergence stop enabled: "
                    f"entropy < {self.stage1_entropy_stop_threshold:.6f}\n"
                )
            if self.run_output_dir:
                f.write(f"[信息] 统一运行输出目录（Unified run output dir）: {self.run_output_dir}\n")


        self.current_episode = 0
        self.total_episodes = self.stage1_rl_episode_limit or PPO_MAX_EPISODES
        self.current_entropy_coef = PPO_ENTROPY_INITIAL
        self.current_lr = self.stage1_ppo_lr_initial


        self.reward_history = deque(maxlen=RUNNING_REWARD_HISTORY_SIZE)
        self.reward_history_sum = 0.0
        self.reward_history_sumsq = 0.0
        self.reward_mean = 0.0
        self.reward_std = 1.0


        self.return_normalizer = RunningMeanStd()


        self.final_eval_config_source = (final_eval_config_source or 'search').lower()
        self.final_eval_config_path = final_eval_config_path or 'glue_final_configs_best_ppo.json'
        self.manual_stage1_gelu = manual_stage1_gelu
        self.manual_stage1_softmax = manual_stage1_softmax
        self.manual_stage2_noise = manual_stage2_noise
        self.stage2_fixed_config_source = str(
            stage2_fixed_config_source or 'all4'
        ).strip().lower()
        self.stage2_fixed_config_path = str(stage2_fixed_config_path or '').strip()
        self.stage2_manual_gelu = stage2_manual_gelu
        self.stage2_manual_softmax = stage2_manual_softmax
        if self.stage2_fixed_config_source not in (
                'all4', 'stage1_result', 'json', 'manual'):
            raise ValueError(
                f"Unsupported stage2_fixed_config_source "
                f"'{self.stage2_fixed_config_source}'. Use one of: "
                "all4, stage1_result, json, manual."
            )
        if self.stage2_fixed_config_source in ('all4', 'stage1_result'):
            if self.stage2_fixed_config_path:
                raise ValueError(
                    f"stage2_fixed_config_source='{self.stage2_fixed_config_source}' "
                    "does not accept stage2_fixed_config_path."
                )
            if self.stage2_manual_gelu is not None or self.stage2_manual_softmax is not None:
                raise ValueError(
                    f"stage2_fixed_config_source='{self.stage2_fixed_config_source}' "
                    "does not accept stage2_manual_gelu/stage2_manual_softmax."
                )
        elif self.stage2_fixed_config_source == 'json':
            if not self.stage2_fixed_config_path:
                raise ValueError(
                    "stage2_fixed_config_source='json' requires "
                    "stage2_fixed_config_path."
                )
            if self.stage2_manual_gelu is not None or self.stage2_manual_softmax is not None:
                raise ValueError(
                    "stage2_fixed_config_source='json' does not accept "
                    "stage2_manual_gelu/stage2_manual_softmax."
                )
        elif self.stage2_manual_gelu is None or self.stage2_manual_softmax is None:
            raise ValueError(
                "stage2_fixed_config_source='manual' requires both "
                "stage2_manual_gelu and stage2_manual_softmax."
            )
        self.final_eval_random_seed = int(final_eval_random_seed)
        self.final_eval_permutation_trials = max(0, int(final_eval_permutation_trials))
        self.final_eval_cost_equivalent_trials = max(0, int(final_eval_cost_equivalent_trials))
        self.final_eval_budget_equivalent_trials = max(0, int(final_eval_budget_equivalent_trials))
        self.final_eval_stage1_budget_trials = max(0, int(final_eval_stage1_budget_trials))
        self.final_eval_stage2_budget_trials = max(0, int(final_eval_stage2_budget_trials))
        self.final_eval_repeat_n = max(1, int(final_eval_repeat_n))
        self.final_eval_preset = str(final_eval_preset or 'default').strip() or 'default'
        self.final_eval_output_root = str(final_eval_output_root or '').strip()
        self.final_eval_run_name = str(final_eval_run_name or '').strip()
        self.final_eval_random_enabled = self._coerce_bool_flag(
            final_eval_random_enabled, 'final_eval_random_enabled')
        self.final_eval_action_config = str(final_eval_action_config or '').strip()
        self.final_eval_action_ranges = final_eval_action_ranges
        self.final_eval_action_fixed = final_eval_action_fixed
        self.final_eval_cost_match_count = max(0, int(final_eval_cost_match_count))
        self.final_eval_cost_match_max_attempts = max(0, int(final_eval_cost_match_max_attempts))
        self.skip_stage1_rl = self._coerce_bool_flag(skip_stage1_rl, 'skip_stage1_rl')
        self.skip_noise_rl = self._coerce_bool_flag(skip_noise_rl, 'skip_noise_rl')
        self.skip_final_eval = self._coerce_bool_flag(skip_final_eval, 'skip_final_eval')
        self.final_eval_only = self._coerce_bool_flag(final_eval_only, 'final_eval_only')
        self.needs_stage2_fixed_config = (not self.skip_noise_rl) or (not self.skip_final_eval)
        self.resume_run_dir = str(resume_run_dir or '').strip()

        if self.final_eval_config_source not in (
                'search', 'json', 'manual', 'max',
                'stage2-max', 'stage2_max', 'blb-max', 'blb_max',
        ):
            raise ValueError(
                f"Unsupported final_eval_config_source '{self.final_eval_config_source}'. "
                "Use one of: search, json, manual, max."
            )


        _has_stage1_manual = (
            self.manual_stage1_gelu is not None
            and self.manual_stage1_softmax is not None
        )
        _has_stage2_manual = self.manual_stage2_noise is not None
        _has_json_fallback = bool(
            self.final_eval_config_path
            and os.path.isfile(self.final_eval_config_path)
        )
        _can_fallback_stage1 = _has_stage1_manual or _has_json_fallback
        _can_fallback_stage2 = _has_stage2_manual or _has_json_fallback

        if (
            self.needs_stage2_fixed_config
            and self.skip_stage1_rl
            and self.final_eval_config_source == 'search'
            and self.stage2_fixed_config_source == 'stage1_result'
            and (not _can_fallback_stage1)
            and (not self.final_eval_only)
        ):
            raise ValueError(
                "skip_stage1_rl=True 且 final_eval_config_source='search' 时，"
                "需要提供 Stage-1 回退配置（json/manual）。"
                "请提供包含 stage1 的 --final-eval-config，"
                "或同时提供 --manual-stage1-gelu 与 --manual-stage1-softmax。"
            )

        if self.skip_stage1_rl and (
            self.stage1_rl_episodes_specified
            or self.stage1_rl_unbounded_until_entropy
            or self.stage1_rl_episodes != PPO_MAX_EPISODES
        ):
            raise ValueError(
                "stage1_rl_episodes was explicitly set, but skip_stage1_rl=True. "
                "Remove --stage1-rl-episodes or disable --skip-stage1-rl."
            )

        if (not self.skip_stage1_rl) and self.final_eval_config_source != 'search':

            print(
                "[自动适配] 启用 Stage-1 搜索，自动将 final_eval_config_source 从 "
                f"'{self.final_eval_config_source}' 切换为 'search'。"
            )
            self.final_eval_config_source = 'search'

        if (
            (not self.skip_stage1_rl)
            and not self.stage1_rl_unbounded_until_entropy
            and self.stage1_rl_episodes < PPO_UPDATE_INTERVAL
        ):
            raise ValueError(
                f"stage1_rl_episodes={self.stage1_rl_episode_limit} is too small. "
                f"It must be >= PPO_UPDATE_INTERVAL ({PPO_UPDATE_INTERVAL}) so Stage-1 PPO can update at least once."
            )

        if self.final_eval_config_source == 'manual':
            if (
                self.manual_stage1_gelu is None
                or self.manual_stage1_softmax is None
                or self.manual_stage2_noise is None
            ):
                raise ValueError(
                    "final_eval_config_source='manual' 时必须同时提供 "
                    "manual_stage1_gelu、manual_stage1_softmax 与 manual_stage2_noise。"
                )

        if self.skip_noise_rl and (not self.skip_final_eval) and self.final_eval_config_source == 'search':
            if (not _can_fallback_stage2) and (not self.final_eval_only):
                raise ValueError(
                    "skip_noise_rl=True 且 final_eval_config_source='search' 时，"
                    "需要提供 Stage-2 回退配置（json/manual）。"
                    "请提供包含 stage2 的 --final-eval-config，"
                    "或提供 --manual-stage2-noise。"
                )

        if (
            self.skip_stage1_rl
            and self.skip_noise_rl
            and (not self.skip_final_eval)
            and self.final_eval_config_source == 'search'
            and (not self.final_eval_only)
        ):
            raise ValueError(
                "当 Stage-1 与 Stage-2 搜索都被跳过时，统一 final-eval 不能使用 "
                "final_eval_config_source='search'。"
            )

        if self.skip_noise_rl and (
            self.stage2_rl_episodes_specified
            or self.stage2_rl_episodes != 0
        ):
            raise ValueError(
                "stage2_rl_episodes was explicitly set, but skip_noise_rl=True. "
                "Remove --stage2-rl-episodes or disable --skip-noise-rl."
            )

        if (not self.skip_noise_rl) and (not self.skip_final_eval) and self.final_eval_config_source != 'search':
            raise ValueError(
                "检测到 Stage-2 噪声 RL 将执行且统一 final-eval 未跳过，但 final_eval_config_source 不是 'search'。"
                "为避免“前面跑 RL、后面却用手动/JSON 配置评估”的流程混用，"
                "执行噪声 RL 且保留 final-eval 时只能使用 search。"
                "若要使用 json/manual，请设置 skip_noise_rl=True。"
            )

        if (
            (not self.skip_noise_rl)
            and str(blb_v3_search_backend or "ppo").strip().lower() == "ppo"
            and self.stage2_rl_episodes != 0
            and self.stage2_rl_episodes < PPO_UPDATE_INTERVAL
        ):
            raise ValueError(
                f"stage2_rl_episodes={self.stage2_rl_episodes} is too small. "
                f"It must be >= PPO_UPDATE_INTERVAL ({PPO_UPDATE_INTERVAL}) so Stage-2 PPO can update at least once."
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
        self.blb_v3_min_convergence_episodes = int(
            blb_v3_min_convergence_episodes
        )
        self.blb_v3_convergence_patience_updates = int(
            blb_v3_convergence_patience_updates
        )
        if self.blb_v3_min_convergence_episodes < 90_000:
            raise ValueError("minimum convergence episode must be at least 90000")
        if self.blb_v3_convergence_patience_updates < 100:
            raise ValueError("convergence patience must be at least 100 updates")

        from blb_stage2_rl.search_baselines import (
            normalize_search_backend,
            validate_comparator_scientific_parameters,
        )

        self.blb_v3_search_backend = normalize_search_backend(
            blb_v3_search_backend
        )
        self.blb_v3_search_evaluation_budget = int(
            blb_v3_search_evaluation_budget
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
        self.blb_v3_search_patience_generations = int(
            blb_v3_search_patience_generations
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
        self.blb_v3_search_full_validation = self._coerce_bool_flag(
            blb_v3_search_full_validation, "blb_v3_search_full_validation"
        )
        self.comparator_smoke = self._coerce_bool_flag(
            comparator_smoke, "comparator_smoke"
        )
        self.comparator_stage1_only = self._coerce_bool_flag(
            comparator_stage1_only, "comparator_stage1_only"
        )
        if self.blb_v3_search_backend == "ppo":
            if self.comparator_smoke or self.comparator_stage1_only:
                raise ValueError("comparator flags require a comparator backend")
        else:
            if self.mrpc_reproducibility is None:
                raise ValueError("comparators require the MRPC reproducibility fixture")
            model_id = str(getattr(getattr(self.model, "config", None), "_name_or_path", ""))
            if (
                int(self.total_layers) != 12
                or str(self.data_path).lower() != "mrpc"
                or model_id.lower() != "textattack/bert-base-uncased-mrpc"
            ):
                raise ValueError(
                    "formal comparators require textattack BERT-base MRPC"
                )
            if int(self.final_eval_random_seed) != 42 or int(self.blb_v3_seed or 42) != 42:
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
            if self.comparator_smoke:
                if (
                    self.blb_v3_search_evaluation_budget != 1
                    or self.blb_v3_search_full_validation
                    or int(self.stage2_k_trials) != 3
                    or not self.skip_final_eval
                ):
                    raise ValueError("comparator smoke contract mismatch")
            else:
                expected_budget = {
                    "bo_rf": 10_000 if self.comparator_stage1_only else 50_000,
                    "greedy": 6**12,
                    "coinn_ga": 11_464,
                }[self.blb_v3_search_backend]
                if self.blb_v3_search_evaluation_budget != expected_budget:
                    raise ValueError("formal comparator evaluation budget mismatch")
            if self.blb_v3_search_backend == "coinn_ga" and (
                self.blb_v3_search_population_size != 64
                or self.blb_v3_search_patience_generations != 5
            ):
                raise ValueError("formal COINN-GA contract mismatch")
            if self.blb_v3_search_backend == "bo_rf" and (
                self.blb_v3_search_initial_design_size != 64
                or self.blb_v3_search_candidate_pool_size != 2_048
                or self.blb_v3_search_patience_generations
                != (1_000 if self.comparator_stage1_only else 2_000)
                or self.blb_v3_search_rf_n_estimators != 128
                or self.blb_v3_search_rf_min_samples_leaf != 2
            ):
                raise ValueError("formal BO-RF contract mismatch")
            if self.comparator_stage1_only:
                if self.skip_stage1_rl or not self.skip_noise_rl or not self.skip_final_eval:
                    raise ValueError("Stage-1-only comparator routing mismatch")
            elif self.skip_stage1_rl or self.skip_noise_rl:
                raise ValueError("two-stage comparator must run both searches")
            if (
                not self.comparator_smoke
                and (
                    not self.blb_v3_search_full_validation
                    or self.blb_v3_final_selection_top_n != 5
                )
            ):
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
            if not self.comparator_smoke and trial_contract != (5, 3, 3, 15, 15):
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

    def _build_final_eval_runner(self):
        return UnifiedFinalEvaluationModule(
            evaluator=self,
            config_source=self.final_eval_config_source,
            config_path=self.final_eval_config_path,
            manual_stage1_gelu=self.manual_stage1_gelu,
            manual_stage1_softmax=self.manual_stage1_softmax,
            manual_stage2_noise=self.manual_stage2_noise,
            random_seed=self.final_eval_random_seed,
            permutation_trials=self.final_eval_permutation_trials,
            cost_equivalent_trials=self.final_eval_cost_equivalent_trials,
            budget_equivalent_trials=self.final_eval_budget_equivalent_trials,
            stage1_budget_trials=self.final_eval_stage1_budget_trials,
            stage2_budget_trials=self.final_eval_stage2_budget_trials,
            repeat_n=self.final_eval_repeat_n,
            results_dir=self.final_eval_dir,
        )

    def _build_stage2_fixed_config_resolver(self):
        source = self.stage2_fixed_config_source
        resolver_source = 'search' if source == 'stage1_result' else source
        return UnifiedFinalEvaluationModule(
            evaluator=self,
            config_source=resolver_source,
            config_path=self.stage2_fixed_config_path,
            manual_stage1_gelu=self.stage2_manual_gelu,
            manual_stage1_softmax=self.stage2_manual_softmax,
            random_seed=self.final_eval_random_seed,
            permutation_trials=0,
            cost_equivalent_trials=0,
            budget_equivalent_trials=0,
            stage1_budget_trials=0,
            stage2_budget_trials=0,
            repeat_n=1,
            results_dir=getattr(self, 'final_eval_dir', None),
        )

    def _maybe_snapshot_decoupled_stage1_record(
        self, *, best_config, base_gelu, base_softmax,
        episode_metric1s, episode_metric2s, episode_losses,
        best_reward, best_cost, completed_episodes,
    ):
        """解耦 stage1-only 完成时：归档 config + 基础指标 + 曲线进 stage1/record/，并打 COMPLETED。

        全程 best-effort：任何异常只记日志，绝不让训练在收尾处崩溃。基础指标来自
        训练中固定的 train_probe；重型同-cost 51 组对比是独立工具。
        """
        try:
            import datetime as _dt
            from rfr.common.config import run_layout as _rl

            wd = os.path.normpath(str(self.run_output_dir or ""))
            if not wd or wd == ".":
                return
            combo = os.path.basename(wd)
            root = os.path.dirname(os.path.dirname(wd))

            def _arr(x, fb):
                try:
                    return [int(v) for v in (x if x is not None else fb)]
                except Exception:
                    return [int(v) for v in fb]

            cfg = best_config if isinstance(best_config, dict) else {}
            gelu = _arr(cfg.get("gelu"), base_gelu)
            softmax = _arr(cfg.get("softmax"), base_softmax)

            def _best(arr, fn):
                try:
                    vals = [float(v) for v in (arr or []) if v is not None and np.isfinite(float(v))]
                    return fn(vals) if vals else None
                except Exception:
                    return None

            m1 = _best(episode_metric1s, max)
            m2 = _best(episode_metric2s, max)
            loss = _best(episode_losses, min)
            gelu_cost = float(sum(GELU_COST.get(int(g), 0.0) for g in gelu))
            softmax_cost = float(sum(SOFTMAX_COST.get(int(s), 0.0) for s in softmax))


            metric_curve_path = ""
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as _plt
                ys = [float(v) for v in (episode_metric1s or []) if v is not None]
                if ys:
                    _plt.figure(figsize=(8, 4))
                    _plt.plot(range(1, len(ys) + 1), ys, lw=1)
                    _plt.xlabel("episode"); _plt.ylabel("metric1 (train_probe)")
                    _plt.title("Stage-1 metric1 curve")
                    metric_curve_path = os.path.join(wd, "stage1_metric_curve.png")
                    _plt.savefig(metric_curve_path, dpi=150); _plt.close()
            except Exception:
                metric_curve_path = ""

            final_config = {
                "stage": 1,
                "combo": combo,
                "gelu_degree_per_layer": gelu,
                "softmax_degree_per_layer": softmax,
                "gelu_cost": gelu_cost,
                "softmax_cost": softmax_cost,
                "total_degree_cost": gelu_cost + softmax_cost,
            }
            final_eval = {
                "source": "training_best_train_probe",
                "note": "basic single-eval snapshot (训练中记录的 train_probe 最优档); "
                        "重型同-cost 51 组对比见独立 final-eval 工具。",
                "metric1": m1,
                "metric2": m2,
                "loss": loss,
                "best_reward": float(best_reward) if best_reward is not None else None,
                "best_cost": float(best_cost) if best_cost is not None else None,
            }
            metadata = {
                "stage": 1,
                "combo": combo,
                "data_path": getattr(self, "data_path", ""),
                "completed_at": _dt.datetime.now().isoformat(),
                "episodes": int(completed_episodes) if completed_episodes is not None else None,
                "stage1_accuracy_tolerance": getattr(self, "stage1_accuracy_tolerance", None),
                "dataset_protocol_hash": self.dataset_protocol_hash,
            }
            report_md = (
                f"# Stage-1 record: {combo}\n\n"
                f"- gelu_degree_per_layer: {gelu}\n"
                f"- softmax_degree_per_layer: {softmax}\n"
                f"- total_degree_cost: {gelu_cost + softmax_cost}\n"
                f"- best train_probe metric1: {m1}\n"
                f"- best_reward: {best_reward}, best_cost: {best_cost}\n"
                f"- episodes: {completed_episodes}\n"
            )
            curve_paths = [
                getattr(self, "stage1_training_curve_path", ""),
                getattr(self, "stage1_entropy_curve_path", ""),
                metric_curve_path,
            ]
            rdir, rid, n = _rl.snapshot_decoupled_record(
                1, combo, wd,
                final_config=final_config,
                final_eval=final_eval,
                metadata=metadata,
                curve_paths=[p for p in curve_paths if p],
                report_md=report_md,
                root=root,
            )
            self.log(f"  [解耦] Stage-1 已归档进 record → {rdir}（COMPLETED 已标记）")
        except Exception as _e:
            self.log(f"  [解耦][警告] Stage-1 record 归档失败（不影响训练结果）：{_e}")

    def _resolve_stage1_degrees_from_record(self):
        """解耦 stage2-only：从 sibling 的 stage1/record/ 读前置 Stage-1 的 gelu/softmax。

        combo 直接来自 ``run_output_dir`` 的 basename（``<root>/stage2/<combo>``），
        stage1 record 根目录为 ``<root>/stage1/record``。返回 ``(gelu, softmax, source)``。
        """
        import json as _json
        from rfr.common.config import run_layout as _rl

        wd = os.path.normpath(str(self.run_output_dir or ""))
        if not wd or wd == ".":
            raise RuntimeError("解耦 stage2 需要 run_output_dir 来定位 stage1/record/。")
        combo = os.path.basename(wd)
        root = os.path.dirname(os.path.dirname(wd))
        rec_root = os.path.join(root, _rl.STAGE1_SUBDIR, _rl.RECORD_SUBDIR)
        run_id_name = self.stage1_run_id or None
        rec_dir = _rl.latest_record_dir_in_root(rec_root, combo, run_id_name=run_id_name)
        if rec_dir is None:
            if run_id_name:
                raise FileNotFoundError(
                    f"指定的 Stage-1 record 不存在：{os.path.join(rec_root, run_id_name)}"
                )
            raise FileNotFoundError(
                f"未找到 combo='{combo}' 的 Stage-1 record（{rec_root}）。"
                "请先运行 Stage-1（--mode stage1-only），或用 --stage2-fixed-config 显式提供 JSON。"
            )
        cfg_path = os.path.join(rec_dir, "final_config.json")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"Stage-1 record 缺少 final_config.json：{cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = _json.load(f)
        gelu_list = cfg.get("gelu_degree_per_layer")
        if not gelu_list:
            raise ValueError(
                f"Stage-1 record 的 final_config.json 缺少 gelu_degree_per_layer：{cfg_path}"
            )
        gelu = np.asarray(gelu_list, dtype=int)
        sm_list = cfg.get("softmax_degree_per_layer") or [FIXED_SOFTMAX_DEGREE] * self.total_layers
        softmax = np.asarray(sm_list, dtype=int)
        self.log(
            f"[Info] Stage-2 从 Stage-1 record 读取前置配置："
            f"{os.path.basename(rec_dir)} (gelu={gelu.tolist()})"
        )
        return gelu, softmax, f"stage1_record:{os.path.basename(rec_dir)}"

    def _resolve_stage2_fixed_stage1_config(self, search_best_config=None):
        if self.stage2_fixed_config_source == 'all4':
            gelu = np.full(self.total_layers, 4, dtype=int)
            softmax = np.full(
                self.total_layers, FIXED_SOFTMAX_DEGREE, dtype=int
            )
            return (
                gelu,
                softmax,
                f"Stage-2 all4 (softmax fixed deg{FIXED_SOFTMAX_DEGREE})",
                'stage2_all4',
            )

        resolver = self._build_stage2_fixed_config_resolver()


        _use_record = (
            self.stage2_fixed_config_source == 'stage1_result'
            and getattr(self, "decoupled_layout", False)
            and search_best_config is None
        )
        if _use_record:
            gelu, softmax, source = self._resolve_stage1_degrees_from_record()
        else:
            gelu, softmax, source = resolver.resolve_stage1_only(
                search_best_stage1=search_best_config,
                total_layers=self.total_layers,
            )
        gelu = np.asarray(gelu, dtype=int).reshape(-1)
        if gelu.size != int(self.total_layers):
            raise ValueError(
                f"Stage-2 GELU vector length {gelu.size} does not match "
                f"num_layers={self.total_layers}."
            )


        del softmax
        softmax = np.full(self.total_layers, FIXED_SOFTMAX_DEGREE, dtype=int)
        label = f"Stage-1 config ({source}; softmax fixed deg{FIXED_SOFTMAX_DEGREE})"
        return gelu, softmax, label, source

    def _detect_task_type(self):
        self.dataset_key = validate_dataset(self.data_path)
        self.dataset_config = DATASET_METRICS_CONFIG[self.dataset_key]
        return False

    def _log_task_type(self):
        """记录任务类型信息"""
        full_names = self.dataset_config['metric_full_names']
        task_type = 'REGRESSION' if self.is_regression else 'CLASSIFICATION'
        print(f"[信息] 数据集（Dataset）'{self.data_path}' 检测为 {task_type} 任务")
        print(f"[信息] 使用指标（Using metrics）: {', '.join(full_names)}")

    def get_metric_names(self) -> Tuple[str, ...]:
        """
        获取当前数据集的完整指标名称（用于日志显示）
        单指标数据集返回 (name,)，双指标返回 (name1, name2)
        """
        full_names = self.dataset_config['metric_full_names']
        if len(full_names) == 1:
            return (full_names[0],)
        return (full_names[0], full_names[1])

    def get_metric_short_names(self) -> Tuple[str, ...]:
        """获取当前数据集的短指标名称（用于表格）"""
        return self.dataset_config['metric_names']

    def get_num_metrics(self) -> int:
        """返回当前数据集的评估指标数量 (1 或 2)"""
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
        """格式化指标字符串，单指标数据集只显示一个指标"""
        names = self.get_metric_names()
        p = f"{prefix}" if prefix else ""
        if self.get_num_metrics() == 1:
            return f"{p}损失（Loss）: {loss:.6f}, {names[0]}: {m1:.6f}"
        return f"{p}损失（Loss）: {loss:.6f}, {names[0]}: {m1:.6f}, {names[1]}: {m2:.6f}"

    def _fmt_constraints(self, limit_loss, limit_p, limit_s):
        """格式化约束字符串"""
        names = self.get_metric_names()
        if self.get_num_metrics() == 1:
            return f"损失（Loss）<={limit_loss:.4f}, {names[0]}>={limit_p:.4f}"
        return f"损失（Loss）<={limit_loss:.4f}, {names[0]}>={limit_p:.4f}, {names[1]}>={limit_s:.4f}"

    def _write_step_info(self, step_info, f):
        """将单步 StepInfo 写入文件"""
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
            new_entropy = PPO_ENTROPY_START
        else:
            tail = (progress - plateau) / max(1.0 - plateau, 1e-8)
            tail = min(max(tail, 0.0), 1.0)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * tail))
            new_entropy = PPO_ENTROPY_END + (
                PPO_ENTROPY_START - PPO_ENTROPY_END
            ) * cosine_factor
        new_entropy = max(new_entropy, GTRXL_ENTROPY_LOWER_BOUND)
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
        """获取当前熵系数（供ppo_update使用）"""
        return self.current_entropy_coef

    def update_reward_statistics(self, episode_reward):
        """
        PPO 7.1: 更新运行时回报统计量
        维护滑动窗口的均值和标准差，用于回报归一化
        """
        episode_reward = float(episode_reward)
        if len(self.reward_history) == self.reward_history.maxlen:
            old_reward = float(self.reward_history[0])
            self.reward_history_sum -= old_reward
            self.reward_history_sumsq -= old_reward * old_reward
        self.reward_history.append(episode_reward)
        self.reward_history_sum += episode_reward
        self.reward_history_sumsq += episode_reward * episode_reward


        if len(self.reward_history) >= RUNNING_REWARD_MIN_SAMPLES:
            n = float(len(self.reward_history))
            self.reward_mean = self.reward_history_sum / n
            variance = max(self.reward_history_sumsq / n - self.reward_mean * self.reward_mean, 0.0)
            self.reward_std = math.sqrt(variance) + RUNNING_REWARD_EPSILON

    def _detect_layer_attribute(self):
        layers = getattr(getattr(self.model, "bert", None), "encoder", None)
        layers = getattr(layers, "layer", None)
        if layers is None or len(layers) == 0:
            raise TypeError("LayerImportanceEvaluator requires a BERT encoder")
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
            _stage1_episode_desc = (
                f"unbounded until entropy < {self.stage1_entropy_stop_threshold:.6f}"
                if self.stage1_rl_unbounded_until_entropy
                else str(self.stage1_rl_episodes)
            )
            f.write(
                "【学习率配置】\n"
                f"  - 一阶段 PPO 学习率（Stage-1 PPO LR）：raw={self.stage1_rl_lr_raw!r} -> "
                f"{self.stage1_ppo_lr_initial:.6g}（{_stage1_lr_mode}）\n"
                f"  - 二阶段 PPO 学习率（Stage-2 PPO LR）：raw={self.stage2_rl_lr_raw!r} -> "
                f"{self.stage2_ppo_lr_initial:.6g}（{_stage2_lr_mode}）\n"
            )
            f.write(
                "【训练轮数配置】\n"
                f"  - 一阶段 RL 回合数（stage-1 episodes）：{_stage1_episode_desc}\n"
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

    def validate_input_noise_scaling_factors(self, scaling_factors):
        arr = np.asarray(scaling_factors, dtype=int)
        if arr.shape != (self.total_layers,):
            raise ValueError(
                f"input_noise_scaling_factors must have shape ({self.total_layers},), "
                f"but got {arr.shape}"
            )
        invalid = self._unsupported_int_values(arr, INPUT_NOISE_ALLOWED_SCALING_FACTORS)
        if invalid:
            raise ValueError(
                f"Unsupported input-noise scaling factors: {invalid}. "
                f"Allowed values: {list(INPUT_NOISE_ALLOWED_SCALING_FACTORS)}"
            )
        return arr

    def validate_weight_noise_scaling_factors(
            self,
            scaling_factors,
            noise_name,
            allowed_values=None,
    ):
        arr = np.asarray(scaling_factors, dtype=int)
        if arr.shape != (self.total_layers,):
            raise ValueError(
                f"{noise_name}_noise_scaling_factors must have shape ({self.total_layers},), "
                f"but got {arr.shape}"
            )
        allowed_values = tuple(allowed_values or WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
        invalid = self._unsupported_int_values(arr, allowed_values)
        if invalid:
            raise ValueError(
                f"Unsupported {noise_name}-noise scaling factors: {invalid}. "
                f"Allowed values: {list(allowed_values)}"
            )
        return arr

    def _apply_weight_noise_configuration(
            self,
            scaling_factors,
            noise_name,
            replace_method_name,
            state_attr,
            allowed_values=None,
    ):
        allowed_values = tuple(allowed_values or WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
        arr = self.validate_weight_noise_scaling_factors(
            scaling_factors,
            noise_name,
            allowed_values=allowed_values,
        )
        handler_layer_name = "model." + self.layers_attribute
        scaling_map = {sf: [] for sf in allowed_values}
        for idx, scaling_factor in enumerate(arr):
            scaling_map[int(scaling_factor)].append(idx)
        replace_method = getattr(self.reversible_handler, replace_method_name)
        for scaling_factor in allowed_values:
            if scaling_map[scaling_factor]:
                replace_method(
                    scaling_map[scaling_factor],
                    handler_layer_name,
                    scaling_factor=scaling_factor,
                    distribution="encoding",
                )
        setattr(self, state_attr, arr.copy())

    def _clear_weight_noise_configuration(
            self,
            restore_method_name,
            state_attr,
            default_value=None,
    ):
        handler_layer_name = "model." + self.layers_attribute
        restore_method = getattr(self.reversible_handler, restore_method_name)
        restore_method(
            list(range(self.total_layers)),
            handler_layer_name,
        )
        setattr(
            self,
            state_attr,
            np.full(
                self.total_layers,
                (
                    WEIGHT_NOISE_DEFAULT_SCALING_FACTOR
                    if default_value is None
                    else int(default_value)
                ),
                dtype=int,
            ),
        )

    def apply_input_noise_configuration(self, input_noise_scaling_factors):
        """Apply layer-wise x-noise scaling factors for the active second-stage RL."""
        arr = self.validate_input_noise_scaling_factors(input_noise_scaling_factors)
        handler_layer_name = "model." + self.layers_attribute
        scaling_map = {sf: [] for sf in INPUT_NOISE_ALLOWED_SCALING_FACTORS}
        for idx, scaling_factor in enumerate(arr):
            scaling_map[int(scaling_factor)].append(idx)
        for scaling_factor in INPUT_NOISE_ALLOWED_SCALING_FACTORS:
            if scaling_map[scaling_factor]:
                self.reversible_handler.replace_layer_input_noise(
                    scaling_map[scaling_factor],
                    handler_layer_name,
                    scaling_factor=scaling_factor,
                    distribution="fresh",
                )
        self.current_input_noise_scaling_factors = arr.copy()

    def clear_input_noise_configuration(self):
        handler_layer_name = "model." + self.layers_attribute
        self.reversible_handler.restore_layer_input_noise(
            list(range(self.total_layers)),
            handler_layer_name,
        )
        self.current_input_noise_scaling_factors = np.full(
            self.total_layers,
            INPUT_NOISE_DEFAULT_SCALING_FACTOR,
            dtype=int,
        )

    def apply_wq_noise_configuration(self, wq_noise_scaling_factors):
        """Apply layer-wise Wq encoding-noise scaling factors."""
        self._apply_weight_noise_configuration(
            wq_noise_scaling_factors,
            noise_name="wq",
            replace_method_name="replace_layer_query_noise",
            state_attr="current_wq_noise_scaling_factors",
        )

    def clear_wq_noise_configuration(self):
        self._clear_weight_noise_configuration(
            restore_method_name="restore_layer_query_noise",
            state_attr="current_wq_noise_scaling_factors",
        )

    def apply_wk_noise_configuration(self, wk_noise_scaling_factors):
        """Apply layer-wise Wk encoding-noise scaling factors."""
        self._apply_weight_noise_configuration(
            wk_noise_scaling_factors,
            noise_name="wk",
            replace_method_name="replace_layer_key_noise",
            state_attr="current_wk_noise_scaling_factors",
        )

    def clear_wk_noise_configuration(self):
        self._clear_weight_noise_configuration(
            restore_method_name="restore_layer_key_noise",
            state_attr="current_wk_noise_scaling_factors",
        )

    def apply_wv_noise_configuration(self, wv_noise_scaling_factors):
        """Apply layer-wise Wv encoding-noise scaling factors."""
        self._apply_weight_noise_configuration(
            wv_noise_scaling_factors,
            noise_name="wv",
            replace_method_name="replace_layer_value_noise",
            state_attr="current_wv_noise_scaling_factors",
        )

    def clear_wv_noise_configuration(self):
        self._clear_weight_noise_configuration(
            restore_method_name="restore_layer_value_noise",
            state_attr="current_wv_noise_scaling_factors",
        )

    def apply_wo_noise_configuration(self, wo_noise_scaling_factors):
        """Apply layer-wise Wo encoding-noise scaling factors."""
        self._apply_weight_noise_configuration(
            wo_noise_scaling_factors,
            noise_name="wo",
            replace_method_name="replace_layer_attention_output_noise",
            state_attr="current_wo_noise_scaling_factors",
        )

    def clear_wo_noise_configuration(self):
        self._clear_weight_noise_configuration(
            restore_method_name="restore_layer_attention_output_noise",
            state_attr="current_wo_noise_scaling_factors",
        )

    def apply_wffn1_noise_configuration(self, wffn1_noise_scaling_factors):
        """Apply layer-wise Wffn1 encoding-noise scaling factors."""
        self._apply_weight_noise_configuration(
            wffn1_noise_scaling_factors,
            noise_name="wffn1",
            replace_method_name="replace_layer_ffn1_noise",
            state_attr="current_wffn1_noise_scaling_factors",
            allowed_values=WFFN1_NOISE_ALLOWED_SCALING_FACTORS,
        )

    def clear_wffn1_noise_configuration(self):
        self._clear_weight_noise_configuration(
            restore_method_name="restore_layer_ffn1_noise",
            state_attr="current_wffn1_noise_scaling_factors",
            default_value=WFFN1_NOISE_DEFAULT_SCALING_FACTOR,
        )

    def apply_wffn2_noise_configuration(self, wffn2_noise_scaling_factors):
        """Apply layer-wise Wffn2 encoding-noise scaling factors."""
        self._apply_weight_noise_configuration(
            wffn2_noise_scaling_factors,
            noise_name="wffn2",
            replace_method_name="replace_layer_ffn2_noise",
            state_attr="current_wffn2_noise_scaling_factors",
        )

    def clear_wffn2_noise_configuration(self):
        self._clear_weight_noise_configuration(
            restore_method_name="restore_layer_ffn2_noise",
            state_attr="current_wffn2_noise_scaling_factors",
        )

    def apply_weight_noise_configuration(
            self,
            wq_noise_scaling_factors=None,
            wk_noise_scaling_factors=None,
            wv_noise_scaling_factors=None,
            wo_noise_scaling_factors=None,
            wffn1_noise_scaling_factors=None,
            wffn2_noise_scaling_factors=None
            ):
        if wq_noise_scaling_factors is not None:
            self.apply_wq_noise_configuration(wq_noise_scaling_factors)
        if wk_noise_scaling_factors is not None:
            self.apply_wk_noise_configuration(wk_noise_scaling_factors)
        if wv_noise_scaling_factors is not None:
            self.apply_wv_noise_configuration(wv_noise_scaling_factors)
        if wo_noise_scaling_factors is not None:
            self.apply_wo_noise_configuration(wo_noise_scaling_factors)
        if wffn1_noise_scaling_factors is not None:
            self.apply_wffn1_noise_configuration(wffn1_noise_scaling_factors)
        if wffn2_noise_scaling_factors is not None:
            self.apply_wffn2_noise_configuration(wffn2_noise_scaling_factors)

    def clear_weight_noise_configuration(self):
        self.clear_wq_noise_configuration()
        self.clear_wk_noise_configuration()
        self.clear_wv_noise_configuration()
        self.clear_wo_noise_configuration()
        self.clear_wffn1_noise_configuration()
        self.clear_wffn2_noise_configuration()

    def validate_softmax_value_noise_scaling_factors(self, scaling_factors, noise_name):
        arr = np.asarray(scaling_factors, dtype=int)
        if arr.shape != (self.total_layers,):
            raise ValueError(
                f"{noise_name}_scaling_factors must have shape ({self.total_layers},), "
                f"but got {arr.shape}"
            )
        invalid = self._unsupported_int_values(
            arr,
            SOFTMAX_VALUE_NOISE_ALLOWED_SCALING_FACTORS,
        )
        if invalid:
            raise ValueError(
                f"Unsupported {noise_name} scaling factors: {invalid}. "
                f"Allowed values: {list(SOFTMAX_VALUE_NOISE_ALLOWED_SCALING_FACTORS)}"
            )
        return arr

    def apply_softmax_value_noise_configuration(
            self,
            softmax_noise_scaling_factors,
            value_noise_scaling_factors,
            ):
        """Apply layer-wise fresh noise for (softmax + e1) @ (V + e2)."""
        softmax_arr = self.validate_softmax_value_noise_scaling_factors(
            softmax_noise_scaling_factors,
            "softmax_value_softmax_noise",
        )
        value_arr = self.validate_softmax_value_noise_scaling_factors(
            value_noise_scaling_factors,
            "softmax_value_v_noise",
        )
        handler_layer_name = "model." + self.layers_attribute
        pair_map = {}
        for idx, (softmax_sf, value_sf) in enumerate(zip(softmax_arr, value_arr)):
            pair_map.setdefault((int(softmax_sf), int(value_sf)), []).append(idx)
        for (softmax_sf, value_sf), layer_indices in sorted(pair_map.items()):
            self.reversible_handler.replace_layer_softmax_value_noise(
                layer_indices,
                handler_layer_name,
                softmax_scaling_factor=softmax_sf,
                value_scaling_factor=value_sf,
                distribution="fresh",
            )
        self.current_softmax_value_softmax_noise_scaling_factors = softmax_arr.copy()
        self.current_softmax_value_v_noise_scaling_factors = value_arr.copy()

    def clear_softmax_value_noise_configuration(self):
        handler_layer_name = "model." + self.layers_attribute
        self.reversible_handler.restore_layer_softmax_value_noise(
            list(range(self.total_layers)),
            handler_layer_name,
        )
        self.current_softmax_value_softmax_noise_scaling_factors = np.full(
            self.total_layers,
            SOFTMAX_VALUE_NOISE_DEFAULT_SCALING_FACTOR,
            dtype=int,
        )
        self.current_softmax_value_v_noise_scaling_factors = np.full(
            self.total_layers,
            SOFTMAX_VALUE_NOISE_DEFAULT_SCALING_FACTOR,
            dtype=int,
        )

    def evaluate_model_with_attention_noise(
            self,
            gelu_degrees,
            softmax_degrees,
            input_noise_scaling_factors=None,
            wq_noise_scaling_factors=None,
            wk_noise_scaling_factors=None,
            wv_noise_scaling_factors=None,
            wo_noise_scaling_factors=None,
            wffn1_noise_scaling_factors=None,
            wffn2_noise_scaling_factors=None,
            use_train=True,
            split=None
            ):
        """Evaluate a fixed GELU/Softmax config with second-stage noise enabled."""
        self.apply_configuration(gelu_degrees, softmax_degrees)
        input_noise_enabled = input_noise_scaling_factors is not None
        weight_noise_enabled = any(
            config is not None
            for config in (
                wq_noise_scaling_factors,
                wk_noise_scaling_factors,
                wv_noise_scaling_factors,
                wo_noise_scaling_factors,
                wffn1_noise_scaling_factors,
                wffn2_noise_scaling_factors,
            )
        )
        if input_noise_enabled:
            self.apply_input_noise_configuration(input_noise_scaling_factors)
        if weight_noise_enabled:
            self.apply_weight_noise_configuration(
                wq_noise_scaling_factors=wq_noise_scaling_factors,
                wk_noise_scaling_factors=wk_noise_scaling_factors,
                wv_noise_scaling_factors=wv_noise_scaling_factors,
                wo_noise_scaling_factors=wo_noise_scaling_factors,
                wffn1_noise_scaling_factors=wffn1_noise_scaling_factors,
                wffn2_noise_scaling_factors=wffn2_noise_scaling_factors,
            )
        split_name = self._resolve_eval_split(use_train=use_train, split=split)
        dataloader = self.dataloaders[split_name]
        try:
            return self._run_evaluation(
                dataloader,
                use_train=(split_name == "train"),
                split_name=split_name,
            )
        finally:
            if weight_noise_enabled:
                self.clear_weight_noise_configuration()
            if input_noise_enabled:
                self.clear_input_noise_configuration()

    def evaluate_model_with_softmax_value_noise(
            self,
            gelu_degrees,
            softmax_degrees,
            softmax_noise_scaling_factors,
            value_noise_scaling_factors,
            input_noise_scaling_factors=None,
            wq_noise_scaling_factors=None,
            wk_noise_scaling_factors=None,
            wv_noise_scaling_factors=None,
            wo_noise_scaling_factors=None,
            wffn1_noise_scaling_factors=None,
            wffn2_noise_scaling_factors=None,
            use_train=True,
            split=None,
            ):
        """Evaluate with attention-product noise: (softmax + e1) @ (V + e2)."""
        self.apply_configuration(gelu_degrees, softmax_degrees)
        input_noise_enabled = input_noise_scaling_factors is not None
        weight_noise_enabled = any(
            config is not None
            for config in (
                wq_noise_scaling_factors,
                wk_noise_scaling_factors,
                wv_noise_scaling_factors,
                wo_noise_scaling_factors,
                wffn1_noise_scaling_factors,
                wffn2_noise_scaling_factors,
            )
        )

        if input_noise_enabled:
            self.apply_input_noise_configuration(input_noise_scaling_factors)
        if weight_noise_enabled:
            self.apply_weight_noise_configuration(
                wq_noise_scaling_factors=wq_noise_scaling_factors,
                wk_noise_scaling_factors=wk_noise_scaling_factors,
                wv_noise_scaling_factors=wv_noise_scaling_factors,
                wo_noise_scaling_factors=wo_noise_scaling_factors,
                wffn1_noise_scaling_factors=wffn1_noise_scaling_factors,
                wffn2_noise_scaling_factors=wffn2_noise_scaling_factors,
            )
        self.apply_softmax_value_noise_configuration(
            softmax_noise_scaling_factors=softmax_noise_scaling_factors,
            value_noise_scaling_factors=value_noise_scaling_factors,
        )

        split_name = self._resolve_eval_split(use_train=use_train, split=split)
        dataloader = self.dataloaders[split_name]
        try:
            return self._run_evaluation(
                dataloader,
                use_train=(split_name == "train"),
                split_name=split_name,
            )
        finally:
            self.clear_softmax_value_noise_configuration()
            if weight_noise_enabled:
                self.clear_weight_noise_configuration()
            if input_noise_enabled:
                self.clear_input_noise_configuration()


    def _stage1_max_scaling_noise_arrays(self):
        max_in = max(INPUT_NOISE_ALLOWED_SCALING_FACTORS)
        max_w = max(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
        max_ffn1 = max(WFFN1_NOISE_ALLOWED_SCALING_FACTORS)
        n = self.total_layers
        return {
            "input_noise_scaling_factors": np.full(n, max_in, dtype=int),
            "wq_noise_scaling_factors": np.full(n, max_w, dtype=int),
            "wk_noise_scaling_factors": np.full(n, max_w, dtype=int),
            "wv_noise_scaling_factors": np.full(n, max_w, dtype=int),
            "wo_noise_scaling_factors": np.full(n, max_w, dtype=int),
            "wffn1_noise_scaling_factors": np.full(n, max_ffn1, dtype=int),
            "wffn2_noise_scaling_factors": np.full(n, max_w, dtype=int),
        }

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

    def evaluate_model_with_attention_noise_repeated(
            self,
            gelu_degrees,
            softmax_degrees,
            input_noise_scaling_factors=None,
            wq_noise_scaling_factors=None,
            wk_noise_scaling_factors=None,
            wv_noise_scaling_factors=None,
            wo_noise_scaling_factors=None,
            wffn1_noise_scaling_factors=None,
            wffn2_noise_scaling_factors=None,
            repeats=1,
            use_train=True,
            split=None,
            random_noise=False,
            ):
        repeats = max(1, int(repeats))
        split_name = self._resolve_eval_split(use_train=use_train, split=split)

        _cpu_rng_state = torch.get_rng_state()
        _cuda_rng_state_all = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        _np_rng_state = np.random.get_state()
        _py_rng_state = random.getstate()

        trials = []
        try:
            base_seed = int(getattr(self, "final_eval_random_seed", 42))
            for trial_idx in range(repeats):


                if random_noise:
                    trial_seed = int.from_bytes(os.urandom(8), "little") & 0x7FFFFFFFFFFFFFFF
                else:
                    trial_seed = base_seed + trial_idx * 1_000_003
                torch.manual_seed(trial_seed)
                np.random.seed(trial_seed % (2**32))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(trial_seed)
                loss, p, s, t = self.evaluate_model_with_attention_noise(
                    gelu_degrees,
                    softmax_degrees,
                    input_noise_scaling_factors=input_noise_scaling_factors,
                    wq_noise_scaling_factors=wq_noise_scaling_factors,
                    wk_noise_scaling_factors=wk_noise_scaling_factors,
                    wv_noise_scaling_factors=wv_noise_scaling_factors,
                    wo_noise_scaling_factors=wo_noise_scaling_factors,
                    wffn1_noise_scaling_factors=wffn1_noise_scaling_factors,
                    wffn2_noise_scaling_factors=wffn2_noise_scaling_factors,
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
        finally:
            torch.set_rng_state(_cpu_rng_state)
            if _cuda_rng_state_all is not None:
                torch.cuda.set_rng_state_all(_cuda_rng_state_all)
            np.random.set_state(_np_rng_state)
            random.setstate(_py_rng_state)

        summary = summarize_eval_trials(trials)
        summary.update({
            "split_name": split_name,
            "trials": trials,
        })
        return summary


    def get_stage1_exact_baseline_configuration(self):
        return (
            np.full(self.total_layers, STAGE1_ORIGINAL_FUNCTION_DEGREE, dtype=int),
            np.full(self.total_layers, STAGE1_ORIGINAL_FUNCTION_DEGREE, dtype=int),
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

    def get_noise_simulated_cost(
            self,
            input_noise_scaling_factors,
            wq_noise_scaling_factors,
            wk_noise_scaling_factors,
            wv_noise_scaling_factors,
            wo_noise_scaling_factors,
            wffn1_noise_scaling_factors,
            wffn2_noise_scaling_factors,
            ):
        breakdown = {
            "x": float(sum(self.INPUT_NOISE_COST_MAP[int(v)] for v in np.asarray(input_noise_scaling_factors, dtype=int))),
            "wq": float(sum(self.WEIGHT_NOISE_COST_MAP[int(v)] for v in np.asarray(wq_noise_scaling_factors, dtype=int))),
            "wk": float(sum(self.WEIGHT_NOISE_COST_MAP[int(v)] for v in np.asarray(wk_noise_scaling_factors, dtype=int))),
            "wv": float(sum(self.WEIGHT_NOISE_COST_MAP[int(v)] for v in np.asarray(wv_noise_scaling_factors, dtype=int))),
            "wo": float(sum(self.WEIGHT_NOISE_COST_MAP[int(v)] for v in np.asarray(wo_noise_scaling_factors, dtype=int))),
            "wffn1": float(sum(self.WFFN1_NOISE_COST_MAP[int(v)] for v in np.asarray(wffn1_noise_scaling_factors, dtype=int))),
            "wffn2": float(sum(self.WEIGHT_NOISE_COST_MAP[int(v)] for v in np.asarray(wffn2_noise_scaling_factors, dtype=int))),
        }
        return float(sum(breakdown.values())), breakdown

    def _get_max_noise_configuration(self):
        return {
            "input_noise_scaling_factors": np.full(self.total_layers, max(INPUT_NOISE_ALLOWED_SCALING_FACTORS), dtype=int),
            "wq_noise_scaling_factors": np.full(self.total_layers, max(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS), dtype=int),
            "wk_noise_scaling_factors": np.full(self.total_layers, max(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS), dtype=int),
            "wv_noise_scaling_factors": np.full(self.total_layers, max(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS), dtype=int),
            "wo_noise_scaling_factors": np.full(self.total_layers, max(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS), dtype=int),
            "wffn1_noise_scaling_factors": np.full(self.total_layers, max(WFFN1_NOISE_ALLOWED_SCALING_FACTORS), dtype=int),
            "wffn2_noise_scaling_factors": np.full(self.total_layers, max(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS), dtype=int),
        }

    def _reset_runtime_ppo_state(self, stage_label='stage1'):
        self.current_episode = 0
        self.current_entropy_coef = PPO_ENTROPY_INITIAL
        if stage_label == 'stage2':
            self.current_lr = self.stage2_ppo_lr_initial
        else:
            self.current_lr = self.stage1_ppo_lr_initial
        self.reward_history = deque(maxlen=RUNNING_REWARD_HISTORY_SIZE)
        self.reward_history_sum = 0.0
        self.reward_history_sumsq = 0.0
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.return_normalizer = RunningMeanStd()

    def _get_stage1_resume_checkpoint_path(self):
        """如果设置了 resume_run_dir，返回 Stage-1 checkpoint 路径；否则返回 None。"""
        from rfr.search.rl.stage1.checkpoint import STAGE1_CHECKPOINT_FILENAME
        if not self.resume_run_dir:
            return None
        path = os.path.join(self.resume_run_dir, "stage1", STAGE1_CHECKPOINT_FILENAME)
        return path if os.path.isfile(path) else None

    def _get_stage2_resume_checkpoint_path(self):
        if not self.resume_run_dir:
            return None
        from rfr.search.rl.stage2.training import (
            BLB_STAGE2_FINAL_CHECKPOINT_FILENAME,
            BLB_STAGE2_LIVE_CHECKPOINT_FILENAME,
        )

        for filename in (
            BLB_STAGE2_FINAL_CHECKPOINT_FILENAME,
            BLB_STAGE2_LIVE_CHECKPOINT_FILENAME,
        ):
            for progress_dir_name in ("stage2_noise", "blb_stage2"):
                path = os.path.join(
                    self.resume_run_dir,
                    progress_dir_name,
                    "progress",
                    filename,
                )
                if os.path.isfile(path):
                    return path
        return None

    def _stage2_final_eval_ineligible_reason(
            self,
            noise_stage_result,
            ):
        """Return why a comparator result cannot enter scientific final-eval."""
        if not isinstance(noise_stage_result, Mapping):
            raise TypeError("Stage-2 search result must be a mapping")
        if self.blb_v3_search_backend == "ppo":
            return None
        if (
                noise_stage_result.get("search_backend")
                != self.blb_v3_search_backend
        ):
            raise RuntimeError(
                "comparator Stage-2 result backend does not match "
                "the active two-stage search"
            )
        if (
                noise_stage_result.get("status") != "completed"
                or noise_stage_result.get("strict_feasible") is not True
        ):
            return "Stage-2 comparator result is not strict-feasible"
        return None

    def _build_stage2_final_eval_handoff(self, noise_stage_result):
        """Detach the Stage-2 best payload before handing it to final-eval."""
        if not isinstance(noise_stage_result, Mapping):
            raise TypeError("Stage-2 search result must be a mapping")
        status = str(noise_stage_result.get("status", ""))
        if status == "smoke_only_complete":
            raise ValueError(
                "smoke-only Stage-2 search cannot be handed to final evaluation; "
                "rerun with the canonical validation_full gate enabled"
            )
        comparator_backends = {"bo_rf", "greedy", "coinn_ga"}
        search_backend = str(
            noise_stage_result.get("search_backend", "") or ""
        ).lower()
        rl_variant = str(noise_stage_result.get("rl_variant", "") or "")
        comparator_variant_prefix = "blb_v3_layerwise_search_"
        comparator_variant_backend = ""
        if rl_variant.startswith(comparator_variant_prefix):
            comparator_variant_backend = rl_variant[
                len(comparator_variant_prefix):
            ]
            if comparator_variant_backend.endswith("_smoke"):
                comparator_variant_backend = comparator_variant_backend[:-6]
        is_comparator = bool(
            search_backend in comparator_backends
            or comparator_variant_backend in comparator_backends
        )
        if is_comparator:
            if search_backend not in comparator_backends:
                raise ValueError(
                    "comparator Stage-2 result is missing its search backend"
                )
            if status != "completed":
                raise ValueError(
                    "only a strict-feasible completed comparator result can be "
                    "handed to scientific final evaluation"
                )
            if noise_stage_result.get("strict_feasible") is not True:
                raise ValueError(
                    "comparator final evaluation requires a strict-feasible "
                    "Stage-2 result"
                )
            if comparator_variant_backend != search_backend:
                raise ValueError(
                    "comparator Stage-2 result has an rl_variant backend mismatch"
                )

        is_layerwise = rl_variant.startswith("blb_v3_layerwise")
        handoff = {}
        best_noise_cfg = noise_stage_result.get("best_noise_config")
        if isinstance(best_noise_cfg, Mapping):
            handoff.update({
                key: np.asarray(value, dtype=int).copy()
                for key, value in best_noise_cfg.items()
                if isinstance(key, str) and key.endswith("scaling_factors")
            })

        best_action = noise_stage_result.get("blb_v3_best_action_vec")
        if is_layerwise and best_action is None:
            raise ValueError("active layerwise Stage-2 result missing action")
        if best_action is not None:
            handoff["blb_v3_best_action_vec"] = np.asarray(
                best_action, dtype=int,
            ).copy()
            result_profile = str(
                noise_stage_result.get("blb_v3_profile") or ""
            )
            evaluator_profile = str(
                getattr(self, "dataset_key", "") or ""
            )
            if is_comparator and (
                    not result_profile or result_profile != evaluator_profile
            ):
                raise ValueError(
                    "comparator Stage-2 result profile does not match the "
                    "final-eval dataset"
                )
            handoff["blb_v3_profile"] = (
                result_profile or evaluator_profile
            )
            if (
                    is_layerwise
                    and "blb_v3_fusion_count_action" not in noise_stage_result
            ):
                raise ValueError(
                    "active layerwise Stage-2 result missing fusion flag"
                )
            fusion_count_action = bool(
                noise_stage_result.get("blb_v3_fusion_count_action", False)
            )
            handoff["blb_v3_fusion_count_action"] = fusion_count_action
            best_group = noise_stage_result.get("blb_v3_best_action_group")
            if fusion_count_action:
                has_options = isinstance(best_group, Mapping) and (
                    isinstance(best_group.get("option_by_step"), Mapping)
                    or isinstance(best_group.get("option_by_graph"), Mapping)
                )
                overrides = (
                    best_group.get("boosted_overrides")
                    if isinstance(best_group, Mapping) else None
                )
                if not has_options or not isinstance(overrides, (list, tuple)):
                    raise ValueError(
                        "fusion-count Stage-2 final-eval requires a reloadable group"
                    )
            if best_group is not None:
                if not isinstance(best_group, Mapping):
                    raise ValueError("blb_v3_best_action_group must be a mapping")
                handoff["blb_v3_best_action_group"] = copy.deepcopy(best_group)

        if is_comparator:
            if handoff.get("blb_v3_fusion_count_action") is not True:
                raise ValueError(
                    "comparator Stage-2 final evaluation requires fusion-count action"
                )
            group = handoff.get("blb_v3_best_action_group")
            action_matrix = (
                group.get("policy_actions")
                if isinstance(group, Mapping) else None
            )
            boosted_overrides = (
                group.get("boosted_overrides")
                if isinstance(group, Mapping) else None
            )
            if not isinstance(action_matrix, (list, tuple)) or not all(
                    isinstance(row, (list, tuple)) for row in action_matrix
            ):
                raise ValueError(
                    "comparator Stage-2 result is missing its selected action matrix"
                )
            if not isinstance(boosted_overrides, (list, tuple)):
                raise ValueError(
                    "comparator Stage-2 result is missing its boosted overrides"
                )
            final_config_fingerprint = str(
                noise_stage_result.get("final_config_fingerprint") or ""
            )
            if (
                    len(final_config_fingerprint) != 64
                    or any(
                        char not in "0123456789abcdef"
                        for char in final_config_fingerprint
                    )
            ):
                raise ValueError(
                    "comparator Stage-2 result has an invalid final config fingerprint"
                )
            handoff.update({
                "status": "completed",
                "search_backend": search_backend,
                "rl_variant": rl_variant,
                "strict_feasible": True,
                "final_config_fingerprint": final_config_fingerprint,
            })

        return handoff or None

    def _load_prior_rl_search_results(self):
        """final_eval_only=True 时，从 resume_run_dir 或 run_output_dir 读取之前 RL 搜索得到的最优配置。

        - Stage-1：从 ``{dir}/stage1/stage1_rl_checkpoint.pt`` 读取 ``best_config``（含 gelu/softmax）。
        - Stage-2：从 ``{dir}/stage2_noise/progress/noise_rl_checkpoint.pt`` 读取
          ``best_noise_config``（含各 *_scaling_factors）。

        仅读取文件、不写入；不调用任何 graceful-stop 接口；不修改 checkpoint 内容。
        因此与续训和优雅停止完全互斥（续训路径只在执行 stage1/stage2 RL 时被触发，
        而 final_eval_only 已强制跳过这两个阶段）。
        返回 ``(stage1_best_dict_or_None, stage2_best_dict_or_None)``。
        """
        from rfr.search.rl.stage1.checkpoint import STAGE1_CHECKPOINT_FILENAME

        candidate_dirs = []
        if self.resume_run_dir:
            candidate_dirs.append(self.resume_run_dir)
        if self.run_output_dir and self.run_output_dir not in candidate_dirs:
            candidate_dirs.append(self.run_output_dir)

        stage1_best = None
        for _dir in candidate_dirs:
            if stage1_best is not None:
                break
            s1_path = os.path.join(_dir, "stage1", STAGE1_CHECKPOINT_FILENAME)
            if os.path.isfile(s1_path):
                try:
                    ckpt = torch.load(s1_path, map_location="cpu", weights_only=False)
                    cfg = ckpt.get("best_config") or ckpt.get("global_best_config")
                    if cfg and "gelu" in cfg and "softmax" in cfg:
                        stage1_best = {
                            "gelu": np.asarray(cfg["gelu"], dtype=int),
                            "softmax": np.asarray(cfg["softmax"], dtype=int),
                        }
                        self.log(f"[final_eval_only] 加载 Stage-1 RL 最优配置: {s1_path}")
                except Exception as exc:
                    self.log(f"[final_eval_only][警告] 读取 {s1_path} 失败: {exc}")

        def _raise_stage2_failures(label, failures):
            details = "; ".join(
                f"{path}: {type(exc).__name__}: {exc}"
                for path, exc in failures
            )
            error = RuntimeError(f"no valid {label} checkpoint found; {details}")
            raise error from failures[-1][1]

        def _is_reloadable_fusion_group(group):
            if not isinstance(group, Mapping):
                return False
            has_options = (
                isinstance(group.get("option_by_step"), Mapping)
                or isinstance(group.get("option_by_graph"), Mapping)
            )
            overrides = group.get("boosted_overrides")
            if not has_options or not isinstance(overrides, (list, tuple)):
                return False
            return all(
                isinstance(row, Mapping)
                and "block_idx" in row
                and "layer_idx" in row
                and isinstance(row.get("field_values"), Mapping)
                for row in overrides
            )

        def _load_blb_stage2_best():
            from rfr.search.rl.stage2.training import (
                BLB_STAGE2_FINAL_CHECKPOINT_FILENAME,
                BLB_STAGE2_LIVE_CHECKPOINT_FILENAME,
            )

            failures = []
            for _dir in candidate_dirs:
                for filename in (
                        BLB_STAGE2_FINAL_CHECKPOINT_FILENAME,
                        BLB_STAGE2_LIVE_CHECKPOINT_FILENAME,
                ):
                    for progress_dir_name in ("stage2_noise", "blb_stage2"):
                        path = os.path.join(
                            _dir, progress_dir_name, "progress", filename,
                        )
                        if not os.path.isfile(path):
                            continue
                        try:
                            checkpoint = torch.load(
                                path, map_location="cpu", weights_only=False,
                            )
                            if not isinstance(checkpoint, Mapping):
                                raise TypeError(
                                    "BLB Stage-2 checkpoint must contain a mapping, "
                                    f"got {type(checkpoint).__name__}"
                                )
                            strict_best = checkpoint.get("strict_best")
                            strict_best = strict_best if isinstance(strict_best, Mapping) else {}
                            best_action = checkpoint.get("best_action")
                            if best_action is None:
                                best_action = checkpoint.get("blb_v3_best_action_vec")
                            if best_action is None:
                                best_action = strict_best.get("full_vector")
                            best_group = checkpoint.get("blb_v3_best_action_group")
                            if best_group is None:
                                best_group = strict_best.get("best_action_group")
                            checkpoint_rl_variant = str(
                                checkpoint.get("rl_variant", "") or ""
                            )
                            is_layerwise = checkpoint_rl_variant.startswith(
                                "blb_v3_layerwise"
                            )
                            comparator_backends = {
                                "bo_rf", "greedy", "coinn_ga",
                            }
                            checkpoint_backend = str(
                                checkpoint.get("search_backend", "") or ""
                            ).lower()
                            variant_prefix = "blb_v3_layerwise_search_"
                            variant_backend = ""
                            if checkpoint_rl_variant.startswith(variant_prefix):
                                variant_backend = checkpoint_rl_variant[
                                    len(variant_prefix):
                                ]
                                if variant_backend.endswith("_smoke"):
                                    variant_backend = variant_backend[:-6]
                            active_backend = str(
                                getattr(self, "blb_v3_search_backend", "ppo")
                                or "ppo"
                            ).lower()
                            is_comparator = bool(
                                active_backend in comparator_backends
                                or checkpoint_backend in comparator_backends
                                or variant_backend in comparator_backends
                            )
                            comparator_contract = None
                            if is_comparator:
                                contract_errors = []
                                if checkpoint.get("status") != "completed":
                                    contract_errors.append("status")
                                if checkpoint.get("strict_feasible") is not True:
                                    contract_errors.append("strict_feasible")
                                if checkpoint_backend not in comparator_backends:
                                    contract_errors.append("search_backend")
                                if variant_backend != checkpoint_backend:
                                    contract_errors.append("rl_variant")
                                if (
                                        active_backend in comparator_backends
                                        and checkpoint_backend != active_backend
                                ):
                                    contract_errors.append("active backend")
                                fingerprint = str(
                                    checkpoint.get("final_config_fingerprint")
                                    or ""
                                )
                                if (
                                        len(fingerprint) != 64
                                        or any(
                                            char not in "0123456789abcdef"
                                            for char in fingerprint
                                        )
                                ):
                                    contract_errors.append(
                                        "final_config_fingerprint"
                                    )
                                action_matrix = (
                                    best_group.get("policy_actions")
                                    if isinstance(best_group, Mapping) else None
                                )
                                if (
                                        not isinstance(action_matrix, (list, tuple))
                                        or not all(
                                            isinstance(row, (list, tuple))
                                            for row in action_matrix
                                        )
                                ):
                                    contract_errors.append("policy_actions")
                                if contract_errors:
                                    raise ValueError(
                                        "completed comparator checkpoint contract "
                                        "is invalid: "
                                        + ", ".join(contract_errors)
                                    )
                                comparator_contract = {
                                    "status": "completed",
                                    "strict_feasible": True,
                                    "search_backend": checkpoint_backend,
                                    "rl_variant": checkpoint_rl_variant,
                                    "final_config_fingerprint": fingerprint,
                                }
                            missing = []
                            if best_action is None:
                                missing.append("action")
                            if is_layerwise:
                                if "blb_v3_fusion_count_action" not in checkpoint:
                                    missing.append("fusion flag")
                                elif not bool(checkpoint["blb_v3_fusion_count_action"]):
                                    missing.append("enabled fusion flag")
                                if not _is_reloadable_fusion_group(best_group):
                                    missing.append("reloadable group")
                            elif best_group is not None and not isinstance(best_group, Mapping):
                                missing.append("valid group mapping")
                            if missing:
                                raise ValueError(
                                    "active layerwise checkpoint missing "
                                    + ", ".join(missing)
                                    if is_layerwise else "BLB checkpoint missing " + ", ".join(missing)
                                )

                            out = {
                                key: np.asarray(value, dtype=int).copy()
                                for key, value in self._get_max_noise_configuration().items()
                                if isinstance(key, str) and key.endswith("scaling_factors")
                            }
                            out["blb_v3_best_action_vec"] = np.asarray(
                                best_action, dtype=int,
                            ).copy()
                            out["blb_v3_profile"] = str(
                                checkpoint.get("profile")
                                or checkpoint.get("blb_v3_profile")
                                or getattr(self, "dataset_key", "")
                                or ""
                            )
                            out["blb_v3_fusion_count_action"] = bool(
                                checkpoint.get("blb_v3_fusion_count_action", False)
                            )
                            if best_group is not None:
                                out["blb_v3_best_action_group"] = copy.deepcopy(best_group)
                            if comparator_contract is not None:
                                out.update(comparator_contract)
                        except Exception as exc:
                            failures.append((path, exc))
                            self.log(
                                f"[final_eval_only][警告] BLB Stage-2 checkpoint "
                                f"不可用，继续查找: {path}: {exc}"
                            )
                            continue
                        self.log(f"[final_eval_only] 加载 BLB Stage-2 最优配置: {path}")
                        return out
            if failures:
                _raise_stage2_failures("BLB Stage-2", failures)
            return None

        stage2_best = _load_blb_stage2_best()

        return stage1_best, stage2_best

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
        """将搜索到的最佳 policy 汇总到 best_policy/ 目录，便于通用 RL 等下游使用。"""
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

    def run_unified_final_eval(self, stage1_search_best=None, stage2_search_best=None,
                               baseline_stage1_gelu=None, baseline_stage1_softmax=None,
                               baseline_noise_tot_c=None,
                               limit_loss=None, limit_p=None, limit_s=None):
        """统一 final-eval 入口：合并 stage1 + stage2 的最终评估。

        ``stage1_search_best`` 形如 ``{'gelu': [...], 'softmax': [...]}``，由 Stage-1 RL/GA 输出。
        ``stage2_search_best`` 为 dict，键为 ``*_scaling_factors``，由 Stage-2 噪声 RL/GA 输出。
        在 ``config_source`` 为 json / manual 时两者可为 None。
        """
        import numpy as np

        if baseline_stage1_gelu is None or baseline_stage1_softmax is None:
            baseline_stage1_gelu, baseline_stage1_softmax = (
                self.get_stage1_exact_baseline_configuration()
            )
        baseline_stage1_gelu = np.asarray(baseline_stage1_gelu, dtype=int)
        baseline_stage1_softmax = np.asarray(baseline_stage1_softmax, dtype=int)

        if baseline_noise_tot_c is None:
            baseline_noise_config = self._get_max_noise_configuration()
            baseline_noise_tot_c, _ = self.get_noise_simulated_cost(**baseline_noise_config)

        if limit_loss is None or limit_p is None or limit_s is None:
            base_loss, base_p, base_s, _ = self.evaluate_model(
                baseline_stage1_gelu,
                baseline_stage1_softmax,
                use_train=False,
                split=self.get_reward_reference_split_name(),
            )
            selection_limits = self.build_constraint_limits_from_metrics(
                base_loss,
                base_p,
                base_s,
            )
            limit_loss = selection_limits["loss"] if limit_loss is None else limit_loss
            limit_p = selection_limits["metric1"] if limit_p is None else limit_p
            limit_s = selection_limits["metric2"] if limit_s is None else limit_s

        if self._should_run_blb_action_final_eval(stage2_search_best):
            from Paean.blb_action_eval import BLBActionFinalEvaluationModule
            blb_random_enabled = bool(
                self.final_eval_random_enabled
                or int(getattr(self, "final_eval_cost_match_count", 0)) > 0
            )
            runner = BLBActionFinalEvaluationModule(
                evaluator=self,
                config_source=self.final_eval_config_source,
                config_path=self.final_eval_config_path,
                manual_stage1_gelu=self.manual_stage1_gelu,
                manual_stage1_softmax=self.manual_stage1_softmax,
                random_seed=self.final_eval_random_seed,
                random_enabled=blb_random_enabled,
                random_count=self._final_eval_random_count(),
                repeat_n=self.final_eval_repeat_n,
                results_dir=self.final_eval_dir,
                action_config_path=self.final_eval_action_config,
                action_ranges=self.final_eval_action_ranges,
                action_fixed=self.final_eval_action_fixed,
                cost_match_count=int(getattr(self, "final_eval_cost_match_count", 50)),
                cost_match_max_attempts=int(getattr(self, "final_eval_cost_match_max_attempts", 5000)),
            )
            return runner.run(
                search_best_stage1=stage1_search_best,
                search_best_stage2=stage2_search_best,
                baseline_stage1_gelu=baseline_stage1_gelu,
                baseline_stage1_softmax=baseline_stage1_softmax,
                baseline_noise_tot_c=float(baseline_noise_tot_c),
                limit_loss=float(limit_loss),
                limit_p=float(limit_p),
                limit_s=float(limit_s),
            )

        runner = self._build_final_eval_runner()
        return runner.run(
            search_best_stage1=stage1_search_best,
            search_best_stage2=stage2_search_best,
            baseline_stage1_gelu=baseline_stage1_gelu,
            baseline_stage1_softmax=baseline_stage1_softmax,
            baseline_noise_tot_c=float(baseline_noise_tot_c),
            limit_loss=float(limit_loss),
            limit_p=float(limit_p),
            limit_s=float(limit_s),
        )

    def _final_eval_random_count(self):
        return int(
            self.final_eval_permutation_trials
            + self.final_eval_cost_equivalent_trials
            + self.final_eval_budget_equivalent_trials
            + self.final_eval_stage1_budget_trials
            + self.final_eval_stage2_budget_trials
        )

    def _should_run_blb_action_final_eval(self, stage2_search_best=None):
        if str(getattr(self, "final_eval_action_config", "") or "").strip():
            return True
        action_ranges = getattr(self, "final_eval_action_ranges", "")
        if action_ranges not in ("", "[]", (), [], None) and str(action_ranges).strip():
            return True
        action_fixed = getattr(self, "final_eval_action_fixed", "")
        if action_fixed not in ("", "[]", (), [], None) and str(action_fixed).strip():
            return True
        if isinstance(stage2_search_best, dict) and stage2_search_best.get("blb_v3_best_action_vec") is not None:
            return True
        return False

    def _run_evaluation(self, dataloader, use_train=False, split_name=None, *,
                        model=None, device=None):
        """在指定模型上运行评估循环（不修改配置），用于无近似对照组等。

        性能优化（不改变任何数值结果）:
          - 每次 forward 前重新确认 model.eval()；model.to(device) 只在首次调用时执行
          - 移除每批次 cuda.synchronize() (仅用于计时, 不影响计算)
          - 移除每次调用的 dummy warmup forward pass (CUDA kernels 已 warmup)
          - 使用 torch.inference_mode() 替代 no_grad() (禁用版本计数, 更快)
          - 在 GPU 传输前提取 labels, 避免 GPU→CPU 往返
          - non_blocking=True + pin_memory 实现异步 CPU→GPU DMA

        ``model`` / ``device`` overrides let the Stage-1 multi-GPU rollout
        runner invoke the same forward loop against a per-worker replica
        without touching ``self.model`` / ``self.device``. Default
        single-GPU behavior is preserved when both are ``None``.
        """


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
        """评估模型，use_train=True时使用训练集，否则使用验证集

        性能优化: 评估结果缓存。由于 model 在 eval 模式冻结参数、dataloader shuffle=False、
        无任何随机性源, 相同 (gelu, softmax, resolved_split) 评估结果必然 bit-identical。
        直接从缓存返回可节省一次完整数据集前向推理, 不改变任何数值结果。

        Returns:
            ``(loss, metric1, metric2, time_ms)`` — metric2 is 0 for
            single-metric datasets.
        """
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
        """将 logits 转换为预测类别"""
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


    def compute_gae(self, rewards, values, dones, gamma=PPO_GAMMA, lam=PPO_LAMBDA):
        """计算广义优势估计 (GAE)"""
        advantages = []
        gae = 0


        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + values

        return advantages, returns

    def compute_gae_batch(self, rewards, values, dones, gamma=PPO_GAMMA, lam=PPO_LAMBDA):
        """Batch GAE over all episodes while staying on the tensor device."""
        advantages = torch.zeros_like(rewards, dtype=torch.float32)
        gae = torch.zeros(rewards.size(0), dtype=torch.float32, device=rewards.device)

        for t in range(rewards.size(1) - 1, -1, -1):
            if t == rewards.size(1) - 1:
                next_value = torch.zeros_like(gae)
            else:
                next_value = values[:, t + 1]
            not_done = 1.0 - dones[:, t]
            delta = rewards[:, t] + gamma * next_value * not_done - values[:, t]
            gae = delta + gamma * lam * not_done * gae
            advantages[:, t] = gae

        returns = advantages + values
        return advantages, returns

    def ppo_update_gtrxl(self, gtrxl_net, optimizer, buffer, device,
                          mini_batch_episodes=GTRXL_MINI_BATCH_EPISODES, entropy_coef=None,
                          ppo_update_step=0):
        """Run one warm-started, entropy-regularized GTrXL PPO update."""
        if entropy_coef is None:
            entropy_coef = self.get_current_entropy_coef()


        if ppo_update_step < GTRXL_WARMUP_STEPS:
            warmup_factor = (ppo_update_step + 1) / GTRXL_WARMUP_STEPS
            current_lr = self.ppo_lr_initial * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr


        _u_seed = (
            int(getattr(self, "final_eval_random_seed", 42))
            ^ (int(ppo_update_step) * 2654435761)
        ) & 0x7FFFFFFFFFFFFFFF
        torch.manual_seed(_u_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_u_seed)

        (cont_features, layer_indices, prev_g_actions,
         actions_g, old_logprobs, rewards, values, dones, gelu_masks) = buffer.get_batch(device)

        n_eps = cont_features.size(0)
        advantages, returns = self.compute_gae_batch(rewards, values, dones)

        adv_flat = advantages.reshape(-1)
        advantages = (advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        self.return_normalizer.update(returns)
        returns_normalized = self.return_normalizer.normalize(returns).to(
            device=device, dtype=torch.float32
        )
        values_normalized = self.return_normalizer.normalize(values).to(
            device=device, dtype=torch.float32
        )

        last_policy_loss_t = None
        last_value_loss_t = None
        last_entropy_t = None

        kl_early_stop = False
        for epoch in range(PPO_K_EPOCHS):
            if kl_early_stop:
                break
            ep_indices = torch.randperm(n_eps, device=device)
            epoch_kl_acc_t = torch.zeros((), device=device)
            epoch_kl_count = 0

            for start in range(0, n_eps, mini_batch_episodes):
                end = min(start + mini_batch_episodes, n_eps)
                mb_idx = ep_indices[start:end]

                mb_cont = cont_features[mb_idx]
                mb_layer = layer_indices[mb_idx]
                mb_prev_g = prev_g_actions[mb_idx]
                mb_act_g = actions_g[mb_idx]
                mb_old_lp = old_logprobs[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ret = returns_normalized[mb_idx]
                mb_old_val = values_normalized[mb_idx]
                mb_gelu_mask = gelu_masks[mb_idx] if gelu_masks is not None else None

                new_logprobs, entropy, new_values_raw = gtrxl_net.evaluate_actions(
                    mb_cont, mb_layer, mb_prev_g, mb_act_g,
                    gelu_mask=mb_gelu_mask
                )

                new_logprobs_flat = new_logprobs.reshape(-1)
                entropy_flat = entropy.reshape(-1)
                new_values_flat = new_values_raw.reshape(-1)
                mb_old_lp_flat = mb_old_lp.reshape(-1)
                mb_adv_flat = mb_adv.reshape(-1)
                mb_ret_flat = mb_ret.reshape(-1)
                mb_old_val_flat = mb_old_val.reshape(-1)

                ratios = torch.exp(new_logprobs_flat - mb_old_lp_flat)
                surr1 = ratios * mb_adv_flat
                surr2 = torch.clamp(ratios, 1 - PPO_EPS_CLIP, 1 + PPO_EPS_CLIP) * mb_adv_flat
                policy_loss = -torch.min(surr1, surr2).mean()

                new_values_norm = (new_values_flat - self.return_normalizer.mean) / self.return_normalizer.std
                value_clipped = mb_old_val_flat + torch.clamp(
                    new_values_norm - mb_old_val_flat,
                    -VALUE_CLIP_RANGE, VALUE_CLIP_RANGE
                )
                huber_loss_fn = nn.HuberLoss(reduction='none', delta=1.0)
                vl_unclipped = huber_loss_fn(new_values_norm, mb_ret_flat)
                vl_clipped = huber_loss_fn(value_clipped, mb_ret_flat)
                value_loss = torch.max(vl_unclipped, vl_clipped).mean()


                mean_entropy = entropy_flat.mean()
                mean_entropy_detached = mean_entropy.detach()
                entropy_deficit = torch.relu(
                    mean_entropy_detached.new_tensor(GTRXL_ENTROPY_LOWER_BOUND)
                    - mean_entropy_detached
                )
                effective_entropy_coef = (
                    entropy_coef
                    + GTRXL_ENTROPY_RECOVERY_MULTIPLIER * entropy_deficit
                )

                entropy_loss = -mean_entropy

                loss = policy_loss + PPO_VALUE_COEF * value_loss + effective_entropy_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(gtrxl_net.parameters(), 0.5)
                optimizer.step()


                with torch.no_grad():
                    approx_kl_t = (mb_old_lp_flat - new_logprobs_flat).mean()
                epoch_kl_acc_t += approx_kl_t.detach()
                epoch_kl_count += 1

                last_policy_loss_t = policy_loss.detach()
                last_value_loss_t = value_loss.detach()
                last_entropy_t = mean_entropy.detach()

            if epoch_kl_count > 0:
                avg_kl_t = epoch_kl_acc_t / float(epoch_kl_count)
                avg_kl = float(avg_kl_t.item())
                if avg_kl > 1.5 * GTRXL_KL_TARGET:
                    kl_early_stop = True

        last_policy_loss = (
            float(last_policy_loss_t.item()) if last_policy_loss_t is not None else 0.0
        )
        last_value_loss = (
            float(last_value_loss_t.item()) if last_value_loss_t is not None else 0.0
        )
        last_entropy = float(last_entropy_t.item()) if last_entropy_t is not None else 0.0
        return last_policy_loss, last_value_loss, last_entropy


    def _stage1_collect_episode_in_worker(self, *, worker, episode_seed):
        """Run one Stage-1 episode entirely on ``worker``'s own GPU.

        Both the GTrXL policy rollout AND the terminal BERT forward run on
        ``worker.device`` using ``worker.gtrxl_replica`` (a per-worker eval-mode
        copy of the central policy, weight-synced each window). There is NO
        shared policy + lock, so workers no longer serialize on policy access —
        that is the speedup. Determinism is preserved: the action sample is
        seeded from the worker's own device RNG with the GLOBAL-episode-index
        ``episode_seed`` from the runner, and CUDA Philox is device-independent,
        so episode ``g`` samples the same action on any GPU / any GPU count.

        Returns an ``EpisodeRollout`` carrying per-step transitions plus
        per-episode summary metrics for the central bookkeeping.
        """
        from rfr.search.rl.stage1.parallel_runner import EpisodeRollout

        device = worker.device
        policy = worker.gtrxl_replica
        if device.type == "cuda":


            torch.cuda.set_device(device)

        env = worker.env
        state = env.reset()


        prev_g_idx = SOS_TOKEN_GELU

        seq_cont_feats = torch.empty(
            (1, self.total_layers, POLICY_CONTINUOUS_DIM),
            dtype=torch.float32,
            device=device,
        )
        seq_layer_indices = torch.empty(
            (1, self.total_layers), dtype=torch.long, device=device,
        )
        seq_prev_g = torch.empty(
            (1, self.total_layers), dtype=torch.long, device=device,
        )
        seq_gelu_masks = torch.empty(
            (1, self.total_layers, int(STAGE1_GELU_ACTION_MASK.size)),
            dtype=torch.bool,
            device=device,
        )
        gelu_mask_np, gelu_mask_t = _get_stage1_gelu_mask_templates(device)
        seq_gelu_masks.copy_(gelu_mask_t.view(1, 1, -1).expand_as(seq_gelu_masks))

        rollout = EpisodeRollout(
            cont_features=[], layer_indices=[], prev_g_actions=[],
            actions_g=[], logprobs=[], rewards=[], values=[],
            dones=[], gelu_masks=[],
            episode_reward=0.0, episode_loss=0.0, episode_metric1=0.0,
            episode_metric2=0.0, episode_cost=0.0,
            gelu_config=[], softmax_config=[], final_config_metrics=None,
            step_infos=[],
        )

        episode_reward = 0.0
        logprob_tensors = []
        value_tensors = []
        gelu_prob_tensors = []
        for step in range(self.total_layers):
            N = self.total_layers
            layer_idx = int(np.argmax(state[0:N]))


            cont_feat_np = env.get_policy_cont_features()
            cont_feat_record = np.asarray(cont_feat_np, dtype=np.float32)

            cont_feat_t = torch.as_tensor(
                cont_feat_record, dtype=torch.float32, device=device
            )

            seq_cont_feats[0, step].copy_(cont_feat_t)
            seq_layer_indices[0, step] = int(layer_idx)
            seq_prev_g[0, step] = int(prev_g_idx)

            full_cont = seq_cont_feats[:, : step + 1, :]
            full_layer = seq_layer_indices[:, : step + 1]
            full_prev_g = seq_prev_g[:, : step + 1]
            full_gelu_mask = seq_gelu_masks[:, : step + 1, :]


            if device.type == "cuda":
                torch.cuda.manual_seed(int(episode_seed) + step)
            else:
                torch.manual_seed(int(episode_seed) + step)
            with torch.no_grad():
                gelu_action, logprob, value, gelu_probs =\
                    policy.get_action_and_logprob(
                        full_cont, full_layer, full_prev_g,
                        return_probs=True, gelu_mask=full_gelu_mask,
                    )


            gelu_action_idx = int(gelu_action.item())
            next_state, reward, done, info = env.step(gelu_action_idx)
            logprob_tensors.append(logprob.detach().reshape(()))
            value_tensors.append(value.detach().reshape(()))
            gelu_prob_tensors.append(gelu_probs.detach())

            rollout.cont_features.append(cont_feat_record)
            rollout.layer_indices.append(layer_idx)
            rollout.prev_g_actions.append(prev_g_idx)
            rollout.actions_g.append(gelu_action_idx)
            rollout.rewards.append(float(reward))
            rollout.dones.append(float(done))
            rollout.gelu_masks.append(gelu_mask_np)
            rollout.step_infos.append({
                "episode_id": -1,
                "layer_index": info["layer_index"],
                "state_vector": state.tolist(),
                "gelu_action_index": gelu_action_idx,
                "gelu_action_degree": int(GELU_MAP[gelu_action_idx]),
                "step_reward": float(reward),
                "done": bool(done),
                "logprob": None,
                "curr_gelu_degree": info["curr_gelu_degree"],
                "curr_softmax_degree": info["curr_softmax_degree"],
                "gelu_prob_dist": None,
                "critic_value": None,
                "accumulated_cost": info["accumulated_cost"],
                "gelu_config": info["gelu_config"],
                "softmax_config": info["softmax_config"],
                "current_lr": None,
                "current_entropy_coef": None,
            })

            prev_g_idx = gelu_action_idx

            episode_reward += reward
            state = next_state

        logprob_values = _stage1_scalar_tensors_to_float_list(logprob_tensors)
        critic_values = _stage1_scalar_tensors_to_float_list(value_tensors)
        gelu_prob_dists = _stage1_prob_tensors_to_nested_lists(gelu_prob_tensors)
        rollout.logprobs.extend(logprob_values)
        rollout.values.extend(critic_values)
        for idx, step_info in enumerate(rollout.step_infos):
            step_info["logprob"] = logprob_values[idx]
            step_info["critic_value"] = critic_values[idx]
            step_info["gelu_prob_dist"] = gelu_prob_dists[idx]

        rollout.episode_reward = float(episode_reward)
        rollout.gelu_config = list(env.gelu_config)
        rollout.softmax_config = list(env.softmax_config)
        rollout.episode_cost = float(env.accumulated_cost)
        if env.current_episode_metrics is not None:
            rollout.episode_loss = float(env.current_episode_metrics["loss"])
            rollout.episode_metric1 = float(env.current_episode_metrics["metric1"])
            rollout.episode_metric2 = float(env.current_episode_metrics["metric2"])
            rollout.final_config_metrics = {
                "loss": float(env.current_episode_metrics["loss"]),
                "metric1": float(env.current_episode_metrics["metric1"]),
                "metric2": float(env.current_episode_metrics["metric2"]),
                "cost": float(env.current_episode_metrics["cost"]),
            }
        return rollout

    def _build_stage1_parallel_runner(self, *, baseline_metrics, base_tot_c,
                                      constraint_limits, eval_split_name,
                                      proxy_prev_metrics):
        """Construct the Stage-1 multi-GPU rollout runner once at training start.

        Returns ``None`` only when ``self.stage1_rl_devices`` is empty (then the
        the single-GPU central loop runs). Pass an explicit device list — even
        a single id like ``0`` — to take the global-seeded rollout path, which
        produces identical results for any GPU count (1, 4, 5, ...).

        Arguments mirror the per-worker env setup the single-GPU path does
        inline in ``on_evaluate`` (baseline_metrics, base_tot_c, constraint
        limits, proxy prev metrics for the differential reward chain).
        """
        from rfr.search.rl.stage1.parallel_runner import (
            build_stage1_parallel_runner,
            parse_device_ids,
        )

        device_ids = parse_device_ids(self.stage1_rl_devices)
        if len(device_ids) < 1:
            return None

        num_metrics = self.get_num_metrics()
        total_layers = self.total_layers
        evaluator_self = self

        def env_factory(model, handler, device, eval_wrapper):
            env = TransformerOptEnv(
                total_layers,
                base_tot_c,
                baseline_metrics,
                eval_wrapper,
                constraint_limits=dict(constraint_limits),
                num_metrics=num_metrics,
            )
            env.prev_episode_metrics = dict(proxy_prev_metrics)
            return env

        def collect_episode(*, worker, episode_seed):
            return evaluator_self._stage1_collect_episode_in_worker(
                worker=worker,
                episode_seed=episode_seed,
            )

        return build_stage1_parallel_runner(
            primary_model=self.model,
            primary_handler=self.reversible_handler,
            evaluator=self,
            env_factory=env_factory,
            collect_episode_fn=collect_episode,
            device_ids=device_ids,
            eval_split_name=eval_split_name,
            log_fn=self.log,
        )

    def on_evaluate(self, args, state, control, **kwargs):
        self.log("\n" + "="*60)
        self.log("开始配置评估（STARTING CONFIGURATION EVALUATION）")
        self.log(f"最终评估配置来源（FINAL_EVAL_CONFIG_SOURCE）={self.final_eval_config_source}")
        if self.skip_stage1_rl:
            self.log("[信息] 第一阶段RL搜索已跳过（--skip-stage1-rl）。")
        self.log("="*60)

        base_gelu, base_softmax = self.get_stage1_exact_baseline_configuration()
        cost_ref_gelu, cost_ref_softmax = self.get_stage1_cost_reference_configuration()
        base_tot_c, base_g_c, base_s_c = self.get_simulated_cost(cost_ref_gelu, cost_ref_softmax)
        num_metrics = self.get_num_metrics()
        base_loss = base_p = base_s = None
        reward_reference_split = self.get_reward_reference_split_name()

        if self.skip_stage1_rl:
            self.log("\n--- 阶段1（Phase 1）: 已跳过（SKIPPED）（--skip-stage1-rl） ---")
            self.log("[信息] 基线建立已跳过，因其仅用于第一阶段RL搜索。")
            if self.run_output_dir:
                update_persistent_metadata_stage(
                    self.run_output_dir, "stage1_search", "skipped")

            if not self.skip_final_eval:
                base_loss, base_p, base_s, _ = self.stage1_final_evaluate(
                    base_gelu,
                    base_softmax,
                    split=reward_reference_split,
                )
                limit_loss = base_loss * (1.0 + self.error_threshold)
                limit_p = base_p * (1.0 - self.correlation_drop_ratio)
                limit_s = base_s * (1.0 - self.correlation_drop_ratio)
        else:


            self.log(
                "\n--- 阶段1（Phase 1）: 建立基线 "
                "(Establishing Baseline on train_probe) ---"
            )
            if reward_reference_split != TRAIN_PROBE_SPLIT:
                raise RuntimeError(
                    "Stage-1 baseline must use train_probe, got "
                    f"{reward_reference_split!r}."
                )


            try:
                from rfr.search.runtime.probe_runner import (
                    enable_cuda_reward_probe_fast_math,
                )
                enable_cuda_reward_probe_fast_math()
                if torch.cuda.is_available():
                    self.log(
                        "  [stage1-eval] TF32 fast matmul enabled"
                        "（与 Stage-2 reward probe 同一设置）"
                    )
            except Exception as _tf32_exc:
                self.log(f"  [stage1-eval] TF32 enable skipped: {_tf32_exc!r}")

            base_loss_val, base_p_val, base_s_val, base_time_val = self.stage1_evaluate(
                base_gelu,
                base_softmax,
                split=reward_reference_split,
            )

            self.log(f"基线指标（Baseline Metrics）（{reward_reference_split}）：")
            self.log(f"  {self._fmt_metrics(base_loss_val, base_p_val, base_s_val)}")
            self.log("  基线配置（Baseline config）: 原始明文 GELU/Softmax（未安装多项式替换）")
            self.log(
                f"  成本参考（Cost reference）: GELU=4 / Softmax=6, "
                f"仿真成本（Sim Cost）={base_tot_c:.2f} "
                f"(GELU成本G={base_g_c:.2f}, Softmax成本S={base_s_c:.2f})"
            )

            base_loss = base_loss_val
            base_p = base_p_val
            base_s = base_s_val


            limit_loss = base_loss * (1.0 + self.error_threshold)
            limit_p = base_p * (1.0 - self.correlation_drop_ratio)
            limit_s = base_s * (1.0 - self.correlation_drop_ratio)

            self.log("约束条件（Constraints）（基于 train_probe）：")
            self.log(f"  {self._fmt_constraints(limit_loss, limit_p, limit_s)}")


        episode_rewards = []
        episode_losses = []
        episode_metric1s = []
        episode_metric2s = []
        episode_entropies = []
        best_config = None
        search_best_config = None
        global_best_config = None
        best_reward = float("-inf")
        best_cost = float("inf")
        stage1_completed_episodes = 0
        stage1_comparator_result = None

        if (
                not self.skip_stage1_rl
                and self.blb_v3_search_backend != "ppo"
        ):
            from dataclasses import replace

            from rfr.common.json_utils import read_json_file
            from stage1_rl.search_baselines import (
                Stage1Constraints,
                stage1_comparator_search_config,
                validate_stage1_comparator_setup,
            )
            from stage1_rl.search_runner import (
                Stage1SearchGracefulStop,
                build_stage1_search_accounting,
                load_completed_search_result,
                run_stage1_search,
            )
            backend = self.blb_v3_search_backend
            if int(num_metrics) != 2:
                raise RuntimeError(
                    "MRPC Stage-1 comparator requires accuracy and weighted_f1"
                )
            stage1_constraints = Stage1Constraints.from_baseline(
                baseline_loss=float(base_loss),
                baseline_metrics=(float(base_p), float(base_s)),
                loss_relative_tolerance=0.001,
                metric_relative_tolerance=0.001,
                metric_names=("accuracy", "weighted_f1"),
            )
            if (
                    not math.isclose(
                        stage1_constraints.loss_max,
                        float(limit_loss),
                        rel_tol=0.0,
                        abs_tol=0.0,
                    )
                    or tuple(stage1_constraints.metric_mins)
                    != (float(limit_p), float(limit_s))
            ):
                raise RuntimeError(
                    "Stage-1 comparator runtime thresholds do not match the "
                    "0.1% MRPC limits"
                )
            stage1_search_config = stage1_comparator_search_config(backend)
            validate_stage1_comparator_setup(
                backend=backend,
                config=stage1_search_config,
                num_layers=int(self.total_layers),
                constraints=stage1_constraints,
            )
            if self.comparator_smoke:
                stage1_search_config = replace(
                    stage1_search_config,
                    evaluation_cap=1,
                    ga_require_full_generations=False,
                )
            stage1_output_dir = os.path.join(
                self.run_output_dir
                or os.path.dirname(self.stage1_step_info_file),
                "stage1_comparator",
                backend,
            )
            preload_path = os.path.join(stage1_output_dir, "checkpoint.json")
            if not os.path.isfile(preload_path):
                preload_path = None
            stage1_comparator_stop_flag_path = os.path.join(
                stage1_output_dir,
                NOISE_STAGE_STOP_FLAG_FILENAME,
            )
            reset_graceful_stop_state()
            consume_stop_flag_file(stage1_comparator_stop_flag_path)
            install_graceful_stop_handler(log_fn=self.log)

            def stage1_comparator_stop_requested():
                return is_graceful_stop_requested(
                    stage1_comparator_stop_flag_path
                )

            self.log(
                f"\n--- 阶段2（Phase 2）: {backend} Stage-1 search "
                "(train_probe) ---"
            )
            self.log(
                "  [优雅停止] 可发送一次 SIGINT，或创建 "
                f"{stage1_comparator_stop_flag_path}；当前候选完成并落盘后退出。"
            )
            try:
                in_memory_stage1_result = run_stage1_search(
                    backend=backend,
                    evaluator=self,
                    num_layers=int(self.total_layers),
                    constraints=stage1_constraints,
                    config=stage1_search_config,
                    output_dir=stage1_output_dir,
                    manifest={
                        "backend": backend,
                        "model": str(getattr(
                            getattr(self.model, "config", None),
                            "_name_or_path",
                            type(self.model).__name__,
                        )),
                        "dataset": str(self.data_path),
                        "split": TRAIN_PROBE_SPLIT,
                        "dataset_protocol_hash": self.dataset_protocol_hash,
                        "comparator_smoke": bool(self.comparator_smoke),
                        "stage1_bound_into_stage2": not self.comparator_stage1_only,
                        "stage2_backend": (
                            None if self.comparator_stage1_only else backend
                        ),
                    },
                    preload_path=preload_path,
                    stop_requested=stage1_comparator_stop_requested,
                )
            except Stage1SearchGracefulStop as stopped:
                consume_stop_flag_file(stage1_comparator_stop_flag_path)
                if self.run_output_dir:
                    update_persistent_metadata_stage(
                        self.run_output_dir,
                        "stage1_search",
                        "in_progress",
                        extra_fields={
                            "backend": backend,
                            "stopped_by": "graceful_stop",
                            "completed_evaluations": int(
                                stopped.observation_count
                            ),
                            "checkpoint_path": preload_path or os.path.join(
                                stage1_output_dir, "checkpoint.json"
                            ),
                        },
                    )
                self.log(
                    "  [优雅停止] Stage-1 comparator checkpoint 已安全落盘；"
                    "下次用相同参数、不带 --fresh 启动即可续跑。"
                )
                raise SystemExit(0)
            finally:
                uninstall_graceful_stop_handler()
            stage1_comparator_result = load_completed_search_result(
                stage1_output_dir
            )
            if (
                    stage1_comparator_result.as_dict()
                    != in_memory_stage1_result.as_dict()
            ):
                raise RuntimeError(
                    "in-memory Stage-1 result does not match its completed "
                    "ordinary artifacts"
                )
            selected_stage1 = stage1_comparator_result.best
            if not selected_stage1.valid:
                raise RuntimeError(
                    "Stage-1 comparator produced no valid configuration"
                )
            if (
                    not self.comparator_smoke
                    and backend == "greedy"
                    and stage1_comparator_result.termination_reason
                    != "verified_local_optimum"
            ):
                raise RuntimeError(
                    "Stage-1 Greedy did not verify every 1-opt and 2-opt "
                    "local optimum"
                )
            if (
                    not self.comparator_smoke
                    and backend == "coinn_ga"
                    and (
                        stage1_comparator_result.termination_reason
                        != "completed_generations"
                        or int(
                            stage1_comparator_result.config.ga_update_generations
                        ) != 200
                        or stage1_comparator_result.config.ga_stop_on_no_improvement
                        or not stage1_comparator_result.config.ga_require_full_generations
                        or stage1_comparator_result.evaluation_count != 11_464
                    )
            ):
                raise RuntimeError(
                    "Stage-1 COINN-GA did not satisfy the 200-generation "
                    "full-run contract and 11,464-inference full-run contract"
                )

            stage1_result_path = os.path.join(
                stage1_output_dir, "result.json",
            )
            from rfr.common.json_utils import stable_json_hash

            with open(stage1_result_path, "rb") as result_handle:
                stage1_result_sha256 = hashlib.sha256(
                    result_handle.read()
                ).hexdigest()
            stage1_manifest = read_json_file(
                os.path.join(stage1_output_dir, "manifest.json")
            )
            if not isinstance(stage1_manifest, Mapping):
                raise RuntimeError("Stage-1 manifest must be a JSON object")
            stage1_selection_payload = {
                "backend": backend,
                "seed": int(stage1_comparator_result.config.seed),
                "action": list(selected_stage1.action),
                "gelu_degrees": list(selected_stage1.gelu_degrees),
                "softmax_degrees": list(selected_stage1.softmax_degrees),
                "num_layers": int(self.total_layers),
                "dataset_protocol_hash": self.dataset_protocol_hash,
            }
            stage1_selection_binding = {
                **stage1_selection_payload,
                "result_path": stage1_result_path,
                "result_sha256": stage1_result_sha256,
                "selection_hash": stable_json_hash(
                    stage1_selection_payload
                ),
            }
            self.stage1_comparator_selection_binding = dict(
                stage1_selection_binding
            )
            best_config = {
                "gelu": np.asarray(
                    selected_stage1.gelu_degrees, dtype=int,
                ),
                "softmax": np.asarray(
                    selected_stage1.softmax_degrees, dtype=int,
                ),
                "feasible": bool(selected_stage1.feasible),
                "selection_status": (
                    "feasible"
                    if selected_stage1.feasible
                    else "least_violating"
                ),
                "backend": backend,
                "dataset_protocol_hash": self.dataset_protocol_hash,
                "evaluation": selected_stage1.as_dict(),
                "result_path": stage1_result_path,
                "selection_binding": stage1_selection_binding,
                "search_accounting": build_stage1_search_accounting(
                    result=stage1_comparator_result,
                    manifest=stage1_manifest,
                ),
            }
            search_best_config = best_config
            global_best_config = best_config
            best_cost = float(selected_stage1.cost)
            stage1_completed_episodes = int(
                stage1_comparator_result.evaluation_count
            )
            if not self.comparator_stage1_only:
                self.stage2_fixed_config_source = "stage1_result"
            if self.run_output_dir:
                update_persistent_metadata_stage(
                    self.run_output_dir,
                    "stage1_search",
                    (
                        "completed"
                        if selected_stage1.feasible
                        else "completed_infeasible"
                    ),
                    extra_fields={
                        "backend": backend,
                        "feasible": bool(selected_stage1.feasible),
                        "evaluation_count": stage1_completed_episodes,
                        "result_path": stage1_result_path,
                    },
                )


        if (
                not self.skip_stage1_rl
                and self.blb_v3_search_backend == "ppo"
        ):
            self.log("\n--- 阶段2（Phase 2）: PPO强化学习训练（PPO Reinforcement Learning Training） ---")

            step_info_details_dir = os.path.join(os.path.dirname(self.step_info_file), "details")
            os.makedirs(step_info_details_dir, exist_ok=True)
            step_info_chunk_file = [None]
            step_info_chunk_idx = [0]
            step_info_is_resuming = [False]


            step_info_chunk_anchor = [0]
            stage1_warning_file = os.path.join(os.path.dirname(self.step_info_file), "warning.txt")
            stage1_prev_avg_reward = [None]
            stage1_warnings = []

            def _get_stage1_chunk_filename(episode_1based):
                anchor = step_info_chunk_anchor[0]
                rel = episode_1based - anchor - 1
                chunk_start = anchor + (rel // STEP_INFO_CHUNK_SIZE) * STEP_INFO_CHUNK_SIZE + 1
                chunk_end = chunk_start + STEP_INFO_CHUNK_SIZE - 1
                return os.path.join(
                    step_info_details_dir,
                    f"ppo_step_info_{chunk_start}-{chunk_end}.txt",
                )

            def _open_stage1_chunk(episode_1based):
                target = _get_stage1_chunk_filename(episode_1based)
                anchor = step_info_chunk_anchor[0]
                new_idx = (episode_1based - anchor - 1) // STEP_INFO_CHUNK_SIZE
                if step_info_chunk_file[0] is not None and step_info_chunk_idx[0] == new_idx:
                    return step_info_chunk_file[0]
                if step_info_chunk_file[0] is not None:
                    step_info_chunk_file[0].close()
                chunk_start = anchor + new_idx * STEP_INFO_CHUNK_SIZE + 1
                chunk_end = chunk_start + STEP_INFO_CHUNK_SIZE - 1

                if step_info_is_resuming[0] and os.path.isfile(target):
                    f = open(target, "a", encoding="utf-8")
                    f.write(f"\n=== [续训 Resume] PPO StepInfo · 回合 {chunk_start}-{chunk_end} ===\n\n")
                    step_info_is_resuming[0] = False
                else:
                    f = open(target, "w", encoding="utf-8")
                    f.write(f"=== PPO每步信息（StepInfo）回合 {chunk_start}-{chunk_end} ===\n\n")
                    step_info_is_resuming[0] = False
                step_info_chunk_file[0] = f
                step_info_chunk_idx[0] = new_idx
                return f


            gtrxl_net = GTrXLStrategyNetwork(
                num_layers=self.total_layers,
                d_model=GTRXL_D_MODEL,
                n_heads=GTRXL_N_HEADS,
                n_gtrxl_layers=GTRXL_N_LAYERS,
                d_ff=GTRXL_D_FF,
                dropout=GTRXL_DROPOUT
            ).to(self.device)


            optimizer = optim.Adam(gtrxl_net.parameters(), lr=self.stage1_ppo_lr_initial)
            gtrxl_ppo_update_count = 0


            baseline_metrics = (base_loss, base_p, base_s)


            class RLEvaluatorWrapper:
                def __init__(wrapper_self, evaluator, split_name):
                    wrapper_self.evaluator = evaluator
                    wrapper_self.split_name = split_name

                def evaluate_model(wrapper_self, gelu_arr, softmax_arr):
                    return wrapper_self.evaluator.stage1_evaluate(
                        gelu_arr,
                        softmax_arr,
                        split=wrapper_self.split_name,
                    )

            online_reward_split = self.get_online_reward_split_name()
            proxy_base_loss, proxy_base_p, proxy_base_s, _ = self.stage1_evaluate(
                base_gelu,
                base_softmax,
                split=online_reward_split,
            )
            self.log(
                f"[信息] 使用 {online_reward_split} 进行在线奖励计算 "
                f"（约束保持在 {reward_reference_split} 上）"
            )
            rl_evaluator = RLEvaluatorWrapper(self, online_reward_split)

            env = TransformerOptEnv(
                self.total_layers,
                base_tot_c,
                baseline_metrics,
                rl_evaluator,
                constraint_limits={
                    "loss": float(limit_loss),
                    "metric1": float(limit_p),
                    "metric2": float(limit_s),
                },
                num_metrics=self.get_num_metrics(),
            )
            env.prev_episode_metrics = {
                "loss": proxy_base_loss,
                "metric1": proxy_base_p,
                "metric2": proxy_base_s,
                "cost": base_tot_c,
            }

            _stage1_model_type = (
                "bert-large" if int(self.total_layers) == 24
                else "bert-base" if int(self.total_layers) == 12
                else f"layers-{int(self.total_layers)}"
            )
            _stage1_run_source = (
                self.run_output_dir
                or os.path.dirname(self.stage1_step_info_file)
                or f"{_stage1_model_type}-{self.data_path}"
            )
            try:
                _stage1_run_id_base = os.path.relpath(_stage1_run_source, os.getcwd())
            except ValueError:
                _stage1_run_id_base = str(_stage1_run_source)
            _stage1_resume_ckpt_path = self._get_stage1_resume_checkpoint_path()
            _stage1_resume_metadata = None
            if _stage1_resume_ckpt_path:
                _stage1_resume_metadata = torch.load(
                    _stage1_resume_ckpt_path,
                    map_location="cpu",
                    weights_only=False,
                )
            _stage1_resume_run_id = (
                _stage1_resume_metadata.get("structured_run_id")
                if isinstance(_stage1_resume_metadata, dict)
                else None
            )
            _stage1_run_id = (
                str(_stage1_resume_run_id)
                if _stage1_resume_run_id
                else make_unique_run_id(_stage1_run_id_base)
            )
            stage1_data_writer = RLDataPointWriter(
                root_dir="rl_training_data_points",
                run_id=_stage1_run_id,
                stage="stage1",
                model_type=_stage1_model_type,
                dataset=str(self.data_path),
            )
            stage1_data_writer.write_manifest({
                "source_run_output_dir": self.run_output_dir,
                "source_data_run_id_base": _stage1_run_id_base,
                "stage1_step_info_file": self.stage1_step_info_file,
                "total_layers": int(self.total_layers),
                "search_algorithm": self.search_algorithm,
                "dataset_protocol_hash": self.dataset_protocol_hash,
                "rl_algo": "ppo",
                "stage1_episodes_requested": (
                    None
                    if self.stage1_rl_unbounded_until_entropy
                    else int(self.stage1_rl_episode_limit)
                ),
                "stage1_unbounded_until_entropy": bool(self.stage1_rl_unbounded_until_entropy),
                "stage1_entropy_stop_threshold": self.stage1_entropy_stop_threshold,
                "ppo_update_interval": int(PPO_UPDATE_INTERVAL),
                "ppo_lr_initial": float(self.stage1_ppo_lr_initial),
                "stage1_rl_devices": self.stage1_rl_devices,
                "random_seed": int(getattr(self, "final_eval_random_seed", 42)),
                "baseline": {
                    "loss": float(base_loss),
                    "metric1": float(base_p),
                    "metric2": float(base_s),
                    "cost_reference": float(base_tot_c),
                    "gelu": np.asarray(base_gelu, dtype=int).tolist(),
                    "softmax": np.asarray(base_softmax, dtype=int).tolist(),
                },
                "constraints": {
                    "loss_max": float(limit_loss),
                    "metric1_min": float(limit_p),
                    "metric2_min": float(limit_s),
                    "stage1_accuracy_tolerance": float(self.error_threshold),
                    "stage1_metric_tolerance": float(self.correlation_drop_ratio),
                },
                "cost_reference": {
                    "gelu": np.asarray(cost_ref_gelu, dtype=int).tolist(),
                    "softmax": np.asarray(cost_ref_softmax, dtype=int).tolist(),
                },
                "action_space": {
                    "gelu_map": GELU_MAP,
                    "gelu_action_mask": STAGE1_GELU_ACTION_MASK.tolist(),
                    "fixed_softmax_degree": int(FIXED_SOFTMAX_DEGREE),
                },
            })
            self.log(
                f"  [data-points] Stage-1 structured RL data → {stage1_data_writer.run_dir}"
            )


            _stage1_parallel_runner = self._build_stage1_parallel_runner(
                baseline_metrics=baseline_metrics,
                base_tot_c=base_tot_c,
                constraint_limits={
                    "loss": float(limit_loss),
                    "metric1": float(limit_p),
                    "metric2": float(limit_s),
                },
                eval_split_name=online_reward_split,
                proxy_prev_metrics={
                    "loss": proxy_base_loss,
                    "metric1": proxy_base_p,
                    "metric2": proxy_base_s,
                    "cost": base_tot_c,
                },
            )
            if _stage1_parallel_runner is not None:
                self.log(
                    f"  [multi-gpu] Stage-1 rollout enabled: "
                    f"workers={_stage1_parallel_runner.num_workers} "
                    f"devices={[str(w.device) for w in _stage1_parallel_runner.workers]} "
                    f"episodes_per_worker={PPO_UPDATE_INTERVAL // _stage1_parallel_runner.num_workers}"
                )
            _stage1_parallel_stash = deque()
            _stage1_parallel_window_t0 = None
            _stage1_parallel_window_idx = None
            _stage1_parallel_collect_seconds = 0.0
            _stage1_parallel_model_forward_seconds = 0.0
            _stage1_parallel_model_forward_calls = 0
            _stage1_parallel_replay_seconds = 0.0
            _stage1_parallel_detail_seconds = 0.0

            buffer = RecurrentRolloutBuffer()


            best_reward = float('-inf')
            best_cost = float('inf')
            window_best_reward = float('-inf')
            window_best_cost = float('inf')
            window_best_config = None

            self.total_episodes = self.stage1_rl_episode_limit or PPO_MAX_EPISODES
            self._reset_runtime_ppo_state()


            from rfr.search.rl.stage1.checkpoint import (
                save_stage1_rl_checkpoint,
                load_stage1_rl_checkpoint,
                STAGE1_CHECKPOINT_FILENAME,
                recover_stage1_detail_files,
                stage1_detail_file_sizes,
            )
            stage1_checkpoint_path = os.path.join(
                os.path.dirname(self.stage1_step_info_file),
                STAGE1_CHECKPOINT_FILENAME,
            )

            stage1_stop_flag_path = os.path.join(
                os.path.dirname(self.stage1_step_info_file),
                NOISE_STAGE_STOP_FLAG_FILENAME,
            )
            reset_graceful_stop_state()
            consume_stop_flag_file(stage1_stop_flag_path)
            install_graceful_stop_handler(log_fn=self.log)
            self.log(
                f"  [优雅停止] 训练期间可按 Ctrl+C 或创建 {stage1_stop_flag_path} "
                f"触发安全停止（在下一回合边界保存 checkpoint 后退出）。"
            )
            stage1_resume_start_episode = 0
            _stage1_cuda_rng_role_registry = None
            if _stage1_resume_ckpt_path:
                _log_rounded_box(
                    self.log,
                    [
                        "断点续训（Resume Stage-1 from checkpoint）",
                        f"加载: {_stage1_resume_ckpt_path}",
                    ],
                )


                ckpt = load_stage1_rl_checkpoint(
                    _stage1_resume_ckpt_path,
                    gtrxl_net,
                    optimizer,
                    device=self.device,
                    expected_dataset_protocol_hash=self.dataset_protocol_hash,
                )
                _stage1_cuda_rng_role_registry = list(
                    ckpt.get("cuda_rng_state_by_role") or ()
                )
                _checkpoint_run_id = ckpt.get("structured_run_id")
                if (
                    _checkpoint_run_id is not None
                    and str(_checkpoint_run_id) != stage1_data_writer.run_id
                ):
                    raise RuntimeError(
                        "Stage-1 checkpoint structured run-id mismatch: "
                        f"{_checkpoint_run_id!r} != {stage1_data_writer.run_id!r}"
                    )
                stage1_data_writer.recover_jsonl_files(
                    ckpt.get("structured_jsonl_sizes")
                )
                recover_stage1_detail_files(
                    step_info_details_dir,
                    ckpt.get("detail_file_sizes"),
                )
                del _stage1_resume_metadata
                stage1_resume_start_episode = int(ckpt["completed_episodes"])
                gtrxl_ppo_update_count = int(ckpt["gtrxl_ppo_update_count"])
                episode_rewards = list(ckpt["episode_rewards"])
                episode_losses = list(ckpt["episode_losses"])
                episode_metric1s = list(ckpt["episode_metric1s"])
                episode_metric2s = list(ckpt["episode_metric2s"])
                episode_entropies = list(ckpt["episode_entropies"])
                best_reward = float(ckpt["best_reward"])
                best_cost = float(ckpt["best_cost"])
                best_config = ckpt.get("best_config")
                search_best_config = None
                global_best_config = None
                window_best_reward = float(ckpt.get("window_best_reward", float('-inf')))
                window_best_cost = float(ckpt.get("window_best_cost", float('inf')))
                window_best_config = ckpt.get("window_best_config")
                stage1_prev_avg_reward[0] = ckpt.get("stage1_prev_avg_reward")
                stage1_warnings = list(ckpt.get("stage1_warnings", []))
                _ev_rt = ckpt.get("ev_runtime_state", {})
                self.reward_history = deque(
                    _ev_rt.get("reward_history", []),
                    maxlen=RUNNING_REWARD_HISTORY_SIZE,
                )
                self.reward_mean = float(_ev_rt.get("reward_mean", 0.0))
                self.reward_std = float(_ev_rt.get("reward_std", 1.0))
                self._rebuild_reward_statistics_accumulators()
                self.current_episode = int(_ev_rt.get("current_episode", stage1_resume_start_episode))

                if "return_normalizer_mean" in _ev_rt:
                    self.return_normalizer.mean = float(_ev_rt["return_normalizer_mean"])
                    self.return_normalizer.var = float(_ev_rt["return_normalizer_var"])
                    self.return_normalizer.count = float(_ev_rt["return_normalizer_count"])
                step_info_is_resuming[0] = True


                step_info_chunk_anchor[0] = stage1_resume_start_episode
                if self.stage1_rl_unbounded_until_entropy:
                    self.log(
                        f"  已恢复至回合 {stage1_resume_start_episode}，"
                        f"将从回合 {stage1_resume_start_episode + 1} 继续训练至 "
                        f"entropy < {self.stage1_entropy_stop_threshold:.6f}"
                    )
                else:
                    self.log(
                        f"  已恢复至回合 {stage1_resume_start_episode}，"
                        f"将从回合 {stage1_resume_start_episode + 1} 继续训练至 {self.stage1_rl_episodes}"
                    )
                if (
                    self.stage1_rl_episode_limit is not None
                    and stage1_resume_start_episode >= self.stage1_rl_episode_limit
                ):
                    self.log(
                        f"  ⚠ checkpoint 已完成 {stage1_resume_start_episode} 回合，"
                        f"目标回合数 {self.stage1_rl_episode_limit} 无需追加训练。"
                    )

            _stage1_rl_t0 = time.time()
            stage1_completed_episodes = int(stage1_resume_start_episode)
            stage1_stop_reason = "max_episodes"
            _stage1_episode_iter = (
                itertools.count(stage1_resume_start_episode)
                if self.stage1_rl_unbounded_until_entropy
                else range(stage1_resume_start_episode, self.stage1_rl_episode_limit)
            )
            for episode in _stage1_episode_iter:

                current_lr, current_entropy = self.update_hyperparameters(optimizer, episode)


                _handled_via_parallel = False
                if _stage1_parallel_runner is not None:
                    if not _stage1_parallel_stash:
                        _window_idx_for_runner = episode // PPO_UPDATE_INTERVAL
                        if self.stage1_rl_unbounded_until_entropy:
                            _window_size = PPO_UPDATE_INTERVAL
                        else:
                            _remaining_total = self.stage1_rl_episode_limit - episode
                            _window_size = min(PPO_UPDATE_INTERVAL, _remaining_total)
                        _stage1_parallel_window_t0 = time.time()
                        _stage1_parallel_window_idx = _window_idx_for_runner
                        _stage1_parallel_replay_seconds = 0.0
                        _stage1_parallel_detail_seconds = 0.0
                        _stage1_parallel_collect_t0 = time.time()


                        _rollouts = _stage1_parallel_runner.run_window(
                            gtrxl_net=gtrxl_net,
                            total_episodes=_window_size,
                            window_idx=_window_idx_for_runner,
                            base_seed=int(getattr(self, "final_eval_random_seed", 42)),
                        )
                        _stage1_parallel_collect_seconds = (
                            time.time() - _stage1_parallel_collect_t0
                        )
                        _stage1_parallel_stash.extend(_rollouts)
                        from rfr.search.rl.stage1.parallel_runner import format_diagnostics_line

                        if _stage1_parallel_runner.last_diagnostics is not None:
                            _stage1_parallel_model_forward_seconds = float(
                                _stage1_parallel_runner.last_diagnostics.model_forward_seconds
                            )
                            _stage1_parallel_model_forward_calls = int(
                                _stage1_parallel_runner.last_diagnostics.model_forward_calls
                            )
                            self.log(
                                "  " + format_diagnostics_line(
                                    _stage1_parallel_runner.last_diagnostics
                                )
                            )
                        self.log(
                            f"  [stage1-rollout] window={_window_idx_for_runner} collected "
                            f"{len(_rollouts)} episodes across "
                            f"{_stage1_parallel_runner.num_workers} workers"
                        )
                        _shared_eval_cache = getattr(
                            self, "_stage1_worker_eval_cache", None
                        )
                        if _shared_eval_cache is not None:
                            self.log(
                                f"  [stage1-rollout] window={_window_idx_for_runner} "
                                + _shared_eval_cache.stats_line()
                            )
                    rollout = _stage1_parallel_stash.popleft()
                    _stage1_parallel_replay_t0 = time.time()

                    buffer.start_episode()
                    for _k in range(len(rollout.actions_g)):
                        buffer.add_step(
                            cont_feat=rollout.cont_features[_k],
                            layer_idx=rollout.layer_indices[_k],
                            prev_g=rollout.prev_g_actions[_k],
                            action_g=rollout.actions_g[_k],
                            logprob=rollout.logprobs[_k],
                            reward=rollout.rewards[_k],
                            value=rollout.values[_k],
                            done=rollout.dones[_k],
                            gelu_mask=rollout.gelu_masks[_k],
                        )
                    buffer.end_episode()
                    episode_reward = float(rollout.episode_reward)
                    step_infos = [dict(_si) for _si in rollout.step_infos]
                    for _so_idx, _si in enumerate(step_infos):
                        _si["step_global"] = episode * self.total_layers + _so_idx
                        _si["episode_id"] = episode
                        _si["current_lr"] = current_lr
                        _si["current_entropy_coef"] = current_entropy


                    env.gelu_config = list(rollout.gelu_config)
                    env.softmax_config = list(rollout.softmax_config)
                    env.accumulated_cost = float(rollout.episode_cost)
                    env.current_episode_metrics = (
                        dict(rollout.final_config_metrics)
                        if rollout.final_config_metrics is not None
                        else None
                    )
                    _stage1_parallel_replay_seconds += (
                        time.time() - _stage1_parallel_replay_t0
                    )
                    _handled_via_parallel = True

                if not _handled_via_parallel:

                    state = env.reset()
                    prev_g_idx = SOS_TOKEN_GELU


                    seq_cont_feats = torch.empty(
                        (1, self.total_layers, POLICY_CONTINUOUS_DIM),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    seq_layer_indices = torch.empty(
                        (1, self.total_layers), dtype=torch.long, device=self.device,
                    )
                    seq_prev_g = torch.empty(
                        (1, self.total_layers), dtype=torch.long, device=self.device,
                    )
                    seq_gelu_masks = torch.empty(
                        (1, self.total_layers, int(STAGE1_GELU_ACTION_MASK.size)),
                        dtype=torch.bool,
                        device=self.device,
                    )
                    gelu_mask_np, gelu_mask_t = _get_stage1_gelu_mask_templates(self.device)
                    seq_gelu_masks.copy_(gelu_mask_t.view(1, 1, -1).expand_as(seq_gelu_masks))

                    episode_reward = 0
                    step_infos = []
                    transition_records = []
                    logprob_tensors = []
                    value_tensors = []
                    gelu_prob_tensors = []
                    buffer.start_episode()

                    for step in range(self.total_layers):


                        N = self.total_layers
                        layer_idx = int(np.argmax(state[0:N]))
                        cont_feat_np = env.get_policy_cont_features()
                        cont_feat_record = np.asarray(cont_feat_np, dtype=np.float32)

                        cont_feat_t = torch.as_tensor(
                            cont_feat_record, dtype=torch.float32, device=self.device
                        )


                        seq_cont_feats[0, step].copy_(cont_feat_t)
                        seq_layer_indices[0, step] = int(layer_idx)
                        seq_prev_g[0, step] = int(prev_g_idx)


                        full_cont = seq_cont_feats[:, : step + 1, :]
                        full_layer = seq_layer_indices[:, : step + 1]
                        full_prev_g = seq_prev_g[:, : step + 1]
                        full_gelu_mask = seq_gelu_masks[:, : step + 1, :]


                        with torch.no_grad():
                            gelu_action, logprob, value, gelu_probs =\
                                gtrxl_net.get_action_and_logprob(
                                    full_cont, full_layer, full_prev_g,
                                    return_probs=True, gelu_mask=full_gelu_mask
                                )


                        gelu_action_idx = int(gelu_action.item())
                        next_state, reward, done, info = env.step(gelu_action_idx)
                        logprob_tensors.append(logprob.detach())
                        value_tensors.append(value.detach())
                        gelu_prob_tensors.append(gelu_probs.detach())


                        step_info = {
                            'step_global': episode * self.total_layers + step,
                            'episode_id': episode,
                            'layer_index': info['layer_index'],
                            'state_vector': state.tolist(),
                            'gelu_action_index': gelu_action_idx,
                            'gelu_action_degree': int(GELU_MAP[gelu_action_idx]),
                            'step_reward': float(reward),
                            'done': bool(done),
                            'logprob': None,
                            'curr_gelu_degree': info['curr_gelu_degree'],
                            'curr_softmax_degree': info['curr_softmax_degree'],
                            'gelu_prob_dist': None,
                            'critic_value': None,
                            'accumulated_cost': info['accumulated_cost'],
                            'gelu_config': info['gelu_config'],
                            'softmax_config': info['softmax_config'],
                            'current_lr': current_lr,
                            'current_entropy_coef': current_entropy
                        }
                        step_infos.append(step_info)

                        transition_records.append(
                            (
                                cont_feat_record,
                                layer_idx,
                                prev_g_idx,
                                gelu_action_idx,
                                reward,
                                float(done),
                                gelu_mask_np,
                            )
                        )


                        prev_g_idx = gelu_action_idx

                        episode_reward += reward
                        state = next_state

                    logprob_values = _stage1_scalar_tensors_to_float_list(logprob_tensors)
                    critic_values = _stage1_scalar_tensors_to_float_list(value_tensors)
                    gelu_prob_dists = _stage1_prob_tensors_to_nested_lists(gelu_prob_tensors)
                    for idx, step_info in enumerate(step_infos):
                        step_info["logprob"] = logprob_values[idx]
                        step_info["critic_value"] = critic_values[idx]
                        step_info["gelu_prob_dist"] = gelu_prob_dists[idx]


                    for idx, (
                        cont_feat_record,
                        layer_idx,
                        prev_g_record,
                        action_g_record,
                        reward_record,
                        done_record,
                        gelu_mask_record,
                    ) in enumerate(transition_records):
                        buffer.add_step(
                            cont_feat=cont_feat_record,
                            layer_idx=layer_idx,
                            prev_g=prev_g_record,
                            action_g=action_g_record,
                            logprob=logprob_values[idx],
                            reward=reward_record,
                            value=critic_values[idx],
                            done=done_record,
                            gelu_mask=gelu_mask_record,
                        )
                    buffer.end_episode()
                episode_rewards.append(episode_reward)
                stage1_completed_episodes = episode + 1


                if hasattr(env, 'current_episode_metrics') and env.current_episode_metrics is not None:
                    episode_losses.append(env.current_episode_metrics['loss'])
                    episode_metric1s.append(env.current_episode_metrics['metric1'])
                    episode_metric2s.append(env.current_episode_metrics['metric2'])
                else:
                    episode_losses.append(base_loss)
                    episode_metric1s.append(base_p)
                    episode_metric2s.append(base_s)


                self.update_reward_statistics(episode_reward)

                _stage1_detail_t0 = time.time()
                chunk_f = _open_stage1_chunk(episode + 1)
                chunk_f.write(f"--- 回合（Episode） {episode + 1} (奖励Reward={episode_reward:.4f}) ---\n")
                for si in step_infos:
                    self._write_step_info(si, chunk_f)
                    chunk_f.write("\n")
                    stage1_data_writer.write_step(si)
                chunk_f.flush()
                if _handled_via_parallel:
                    _stage1_parallel_detail_seconds += (
                        time.time() - _stage1_detail_t0
                    )


                final_config = {
                    'gelu': np.array(env.gelu_config),
                    'softmax': np.array(env.softmax_config),
                    'cost': env.accumulated_cost,
                    'reward': episode_reward,
                    'dataset_protocol_hash': self.dataset_protocol_hash,
                }

                if episode_reward > window_best_reward:
                    window_best_reward = episode_reward
                    window_best_cost = env.accumulated_cost
                    window_best_config = {
                        'gelu': np.array(env.gelu_config),
                        'softmax': np.array(env.softmax_config),
                        'cost': env.accumulated_cost,
                        'reward': episode_reward,
                        'dataset_protocol_hash': self.dataset_protocol_hash,
                    }

                _stage1_is_new_best = bool(episode_reward > best_reward)
                if _stage1_is_new_best:
                    best_reward = episode_reward
                    best_cost = env.accumulated_cost
                    best_config = final_config.copy()
                    self.log(f"  回合（Episode） {episode+1}: 新最优！（New Best!） 奖励（Reward）={episode_reward:.4f}, 成本（Cost）={env.accumulated_cost:.2f}")
                    self.log(f"    GELU配置: {env.gelu_config}")
                    self.log(f"    Softmax配置: {env.softmax_config}")

                _episode_metrics = (
                    dict(env.current_episode_metrics)
                    if getattr(env, "current_episode_metrics", None) is not None
                    else {
                        "loss": base_loss,
                        "metric1": base_p,
                        "metric2": base_s,
                        "cost": env.accumulated_cost,
                    }
                )
                stage1_data_writer.write_episode({
                    "episode": int(episode + 1),
                    "episode_zero_based": int(episode),
                    "split": online_reward_split,
                    "dataset_protocol_hash": self.dataset_protocol_hash,
                    "reward": float(episode_reward),
                    "loss": float(_episode_metrics["loss"]),
                    "metric1": float(_episode_metrics["metric1"]),
                    "metric2": float(_episode_metrics["metric2"]),
                    "cost": float(env.accumulated_cost),
                    "gelu": np.asarray(env.gelu_config, dtype=int).tolist(),
                    "softmax": np.asarray(env.softmax_config, dtype=int).tolist(),
                    "current_lr": float(current_lr),
                    "current_entropy_coef": float(current_entropy),
                    "reward_mean": float(self.reward_mean),
                    "reward_std": float(self.reward_std),
                    "is_new_best": _stage1_is_new_best,
                    "best_reward_so_far": float(best_reward),
                    "best_cost_so_far": float(best_cost),
                    "best_gelu_so_far": (
                        np.asarray(best_config["gelu"], dtype=int).tolist()
                        if best_config is not None else None
                    ),
                    "best_softmax_so_far": (
                        np.asarray(best_config["softmax"], dtype=int).tolist()
                        if best_config is not None else None
                    ),
                })


                if (episode + 1) % PPO_UPDATE_INTERVAL == 0:
                    _stage1_ppo_update_t0 = time.time()
                    try:
                        policy_loss, value_loss, entropy = self.ppo_update_gtrxl(
                            gtrxl_net, optimizer, buffer, self.device,
                            entropy_coef=current_entropy,
                            ppo_update_step=gtrxl_ppo_update_count
                        )
                    except ElasticGPUFailure:
                        raise
                    except Exception as exc:
                        if not is_recoverable_gpu_failure(exc):
                            raise
                        raise ElasticGPUFailure(
                            device=self.device,
                            role="learner-primary",
                            operation="stage1_ppo_update",
                            cause=exc,
                        ) from exc
                    _stage1_ppo_update_seconds = time.time() - _stage1_ppo_update_t0
                    gtrxl_ppo_update_count += 1
                    buffer.clear()
                    episode_entropies.append(entropy)
                    stage1_entropy_converged = (
                        self.stage1_entropy_stop_threshold is not None
                        and float(entropy) < self.stage1_entropy_stop_threshold
                    )
                    _stage1_parallel_update_payload = None
                    if (
                            _stage1_parallel_runner is not None
                            and _stage1_parallel_window_t0 is not None
                            and _stage1_parallel_window_idx is not None):
                        _diag = _stage1_parallel_runner.last_diagnostics
                        _stage1_parallel_total_seconds = (
                            time.time() - _stage1_parallel_window_t0
                        )
                        _stage1_parallel_window_episodes = (
                            int(sum(_diag.per_worker_episode_counts))
                            if _diag is not None
                            else PPO_UPDATE_INTERVAL
                        )
                        _stage1_parallel_known_seconds = (
                            _stage1_parallel_collect_seconds
                            + _stage1_parallel_replay_seconds
                            + _stage1_parallel_detail_seconds
                            + _stage1_ppo_update_seconds
                        )
                        _stage1_parallel_other_seconds = max(
                            0.0,
                            _stage1_parallel_total_seconds - _stage1_parallel_known_seconds,
                        )
                        _stage1_parallel_ep_per_hour = (
                            _stage1_parallel_window_episodes
                            / max(_stage1_parallel_total_seconds, 1e-9)
                            * 3600.0
                        )
                        _stage1_parallel_update_payload = {
                            "window": int(_stage1_parallel_window_idx),
                            "episodes": int(_stage1_parallel_window_episodes),
                            "total_seconds": float(_stage1_parallel_total_seconds),
                            "collect_seconds": float(_stage1_parallel_collect_seconds),
                            "model_forward_seconds": float(_stage1_parallel_model_forward_seconds),
                            "model_forward_calls": int(_stage1_parallel_model_forward_calls),
                            "replay_seconds": float(_stage1_parallel_replay_seconds),
                            "detail_seconds": float(_stage1_parallel_detail_seconds),
                            "report_write_seconds": float(_stage1_parallel_detail_seconds),
                            "ppo_update_seconds": float(_stage1_ppo_update_seconds),
                            "other_seconds": float(_stage1_parallel_other_seconds),
                            "throughput_ep_per_hour": float(_stage1_parallel_ep_per_hour),
                        }
                        if _diag is not None:
                            _stage1_parallel_update_payload["worker_seconds"] = list(_diag.per_worker_seconds)
                            _stage1_parallel_update_payload["worker_episode_counts"] = list(_diag.per_worker_episode_counts)
                            _stage1_parallel_update_payload["devices"] = list(_diag.devices)
                            _stage1_parallel_update_payload["speedup_vs_sequential"] = float(_diag.speedup_vs_sequential)
                        self.log(
                            "  [stage1-rollout-total] "
                            f"window={_stage1_parallel_window_idx} "
                            f"episodes={_stage1_parallel_window_episodes} "
                            f"total={_stage1_parallel_total_seconds:.3f}s "
                            f"collect={_stage1_parallel_collect_seconds:.3f}s "
                            f"model_forward={_stage1_parallel_model_forward_seconds:.3f}s "
                            f"replay={_stage1_parallel_replay_seconds:.3f}s "
                            f"detail={_stage1_parallel_detail_seconds:.3f}s "
                            f"report_write={_stage1_parallel_detail_seconds:.3f}s "
                            f"ppo_update={_stage1_ppo_update_seconds:.3f}s "
                            f"other={_stage1_parallel_other_seconds:.3f}s "
                            f"throughput={_stage1_parallel_ep_per_hour:.1f}ep/h"
                        )
                        _stage1_parallel_window_t0 = None
                        _stage1_parallel_window_idx = None
                        _stage1_parallel_collect_seconds = 0.0
                        _stage1_parallel_model_forward_seconds = 0.0
                        _stage1_parallel_model_forward_calls = 0
                        _stage1_parallel_replay_seconds = 0.0
                        _stage1_parallel_detail_seconds = 0.0

                    avg_reward = np.mean(episode_rewards[-PPO_UPDATE_INTERVAL:])
                    stage1_data_writer.write_ppo_update({
                        "update": int(gtrxl_ppo_update_count),
                        "episode": int(episode + 1),
                        "policy_loss": float(policy_loss),
                        "value_loss": float(value_loss),
                        "entropy": float(entropy),
                        "avg_reward": float(avg_reward),
                        "current_lr": float(optimizer.param_groups[0]["lr"]),
                        "current_entropy_coef": float(current_entropy),
                        "best_reward_so_far": float(best_reward),
                        "best_cost_so_far": float(best_cost),
                        "buffer_episodes": int(PPO_UPDATE_INTERVAL),
                        "parallel": _stage1_parallel_update_payload,
                    })
                    warmup_status = "warmup" if gtrxl_ppo_update_count <= GTRXL_WARMUP_STEPS else "normal"
                    _log_rounded_box(
                        self.log,
                        [
                            f"回合（Episode） {episode + 1}",
                            f"平均奖励: {avg_reward:.4f}, 策略损失: {policy_loss:.4f}, 价值损失: {value_loss:.4f}, 熵: {entropy:.4f}",
                            (
                                f"[GTrXL调度] LR: {optimizer.param_groups[0]['lr']:.6f}, "
                                f"熵系数: {current_entropy:.6f}, 更新次数: #{gtrxl_ppo_update_count} ({warmup_status})"
                            ),
                        ],
                    )


                    _stage1_reached_episode_cap = (
                        self.stage1_rl_episode_limit is not None
                        and episode + 1 >= self.stage1_rl_episode_limit
                    )
                    if (gtrxl_ppo_update_count % NOISE_RL_PROGRESS_BOX_PPO_INTERVAL == 0
                            or _stage1_reached_episode_cap):
                        _s1_elapsed = time.time() - _stage1_rl_t0
                        _s1_done = episode + 1 - stage1_resume_start_episode
                        _s1_avg_ep = _s1_elapsed / max(_s1_done, 1)
                        _s1_best_lines = []
                        if best_config is not None:
                            _s1_best_lines.append(
                                f"Reward-Best: {best_config.get('reward', 0):.4f}  "
                                f"成本: {best_config.get('cost', 0):.2f}"
                            )
                            _s1_best_lines.append(f"  GELU:    {list(best_config.get('gelu', []))}")
                            _s1_best_lines.append(f"  Softmax: {list(best_config.get('softmax', []))}")
                        else:
                            _s1_best_lines.append("Reward-Best: 尚未找到")
                        if self.stage1_rl_unbounded_until_entropy:
                            _s1_progress_title = (
                                f"Stage-1 RL 进度 · 回合 {episode + 1} / entropy<"
                                f"{self.stage1_entropy_stop_threshold:.4f}"
                            )
                            _s1_progress_lines = [
                                "进度: unbounded until entropy convergence",
                                *_s1_best_lines,
                                f"已用时: {_fmt_elapsed(_s1_elapsed)}  "
                                f"平均每回合: {_fmt_elapsed(_s1_avg_ep)}  "
                                f"PPO 更新: {gtrxl_ppo_update_count} 次",
                            ]
                        else:
                            _s1_remain = self.stage1_rl_episode_limit - (episode + 1)
                            _s1_eta = _s1_avg_ep * _s1_remain
                            _s1_progress_title = (
                                f"Stage-1 RL 进度 · 回合 {episode + 1} / {self.stage1_rl_episode_limit}"
                            )
                            _s1_progress_lines = [
                                _progress_bar(episode + 1, self.stage1_rl_episode_limit),
                                *_s1_best_lines,
                                f"已用时: {_fmt_elapsed(_s1_elapsed)}  "
                                f"预计剩余: {'until entropy stop' if _s1_eta is None else _fmt_elapsed(_s1_eta)}  "
                                f"预计完成: {'until entropy stop' if _s1_eta is None else _fmt_eta_finish(_s1_eta)}  "
                                f"PPO 更新: {gtrxl_ppo_update_count} 次",
                            ]
                        _log_rounded_box(
                            self.log,
                            [
                                _s1_progress_title,
                                *_s1_progress_lines,
                            ],
                            indent="  ",
                        )

                    if stage1_prev_avg_reward[0] is not None:
                        reward_drop = stage1_prev_avg_reward[0] - avg_reward
                        if reward_drop > REWARD_DROP_WARNING_THRESHOLD:
                            window_start_ep = episode + 1 - PPO_UPDATE_INTERVAL + 1
                            window_end_ep = episode + 1
                            involved_files = sorted(set(
                                _get_stage1_chunk_filename(e)
                                for e in range(window_start_ep, window_end_ep + 1)
                            ))
                            involved_basenames = [os.path.basename(fp) for fp in involved_files]
                            warn_msg = {
                                "type": "阶段1奖励骤降",
                                "window": gtrxl_ppo_update_count,
                                "prev_avg": float(stage1_prev_avg_reward[0]),
                                "curr_avg": float(avg_reward),
                                "drop": float(reward_drop),
                                "threshold": float(REWARD_DROP_WARNING_THRESHOLD),
                                "episode_range": (window_start_ep, window_end_ep),
                                "detail_files": involved_basenames,
                            }
                            stage1_warnings.append(warn_msg)
                            self.log(
                                f"  ⚠ 警告: 平均奖励下降 {reward_drop:.4f} "
                                f"(阈值={REWARD_DROP_WARNING_THRESHOLD}), "
                                f"涉及回合 {window_start_ep}-{window_end_ep}"
                            )
                    stage1_prev_avg_reward[0] = avg_reward

                    window_best_reward = float('-inf')
                    window_best_cost = float('inf')
                    window_best_config = None

                    if (
                            not stage1_entropy_converged
                            and (
                                self.stage1_rl_unbounded_until_entropy
                                or (episode + 1) < self.stage1_rl_episode_limit
                            )
                    ):
                        env.prev_episode_metrics = {
                            "loss": proxy_base_loss,
                            "metric1": proxy_base_p,
                            "metric2": proxy_base_s,
                            "cost": base_tot_c,
                        }
                        env.current_episode_metrics = None


                    if step_info_chunk_file[0] is not None:
                        step_info_chunk_file[0].flush()
                    _stage1_structured_jsonl_sizes = (
                        stage1_data_writer.committed_jsonl_sizes()
                    )
                    _stage1_detail_file_sizes = stage1_detail_file_sizes(step_info_details_dir)
                    save_stage1_rl_checkpoint(
                        path=stage1_checkpoint_path,
                        gtrxl_net=gtrxl_net,
                        optimizer=optimizer,
                        episode=episode,
                        gtrxl_ppo_update_count=gtrxl_ppo_update_count,
                        episode_rewards=episode_rewards,
                        episode_losses=episode_losses,
                        episode_metric1s=episode_metric1s,
                        episode_metric2s=episode_metric2s,
                        episode_entropies=episode_entropies,
                        best_reward=best_reward,
                        best_cost=best_cost,
                        best_config=best_config,
                        search_best_config=search_best_config,
                        global_best_config=global_best_config,
                        window_best_reward=window_best_reward,
                        window_best_cost=window_best_cost,
                        window_best_config=window_best_config,
                        ev_runtime_state={
                            "reward_history": list(self.reward_history),
                            "reward_mean": float(self.reward_mean),
                            "reward_std": float(self.reward_std),
                            "current_episode": int(self.current_episode),
                            "return_normalizer_mean": float(self.return_normalizer.mean),
                            "return_normalizer_var": float(self.return_normalizer.var),
                            "return_normalizer_count": float(self.return_normalizer.count),
                        },
                        stage1_prev_avg_reward=stage1_prev_avg_reward[0],
                        stage1_warnings=stage1_warnings,
                        dataset_protocol_hash=self.dataset_protocol_hash,
                        structured_run_id=stage1_data_writer.run_id,
                        structured_jsonl_sizes=_stage1_structured_jsonl_sizes,
                        detail_file_sizes=_stage1_detail_file_sizes,
                        cuda_rng_role_registry=_stage1_cuda_rng_role_registry,
                    )
                    if (
                        not stage1_entropy_converged
                        and not _stage1_reached_episode_cap
                    ):
                        if _stage1_parallel_runner is not None:
                            _deferred_gpu_failure = (
                                _stage1_parallel_runner.pop_deferred_gpu_failure()
                            )
                            if _deferred_gpu_failure is not None:
                                raise _deferred_gpu_failure
                        raise_if_elastic_gpu_restart_requested()
                    if (
                        self.stage1_entropy_stop_threshold is not None
                        and entropy <= self.stage1_entropy_stop_threshold
                    ):
                        stage1_entropy_converged = True
                        stage1_stop_reason = "entropy_converged"
                        self.log(
                            "Stage-1 entropy convergence reached: "
                            f"entropy={entropy:.6f} <= threshold={self.stage1_entropy_stop_threshold:.6f} "
                            f"at episode {episode + 1}"
                        )
                        break

                    if stage1_entropy_converged:
                        stage1_stop_reason = "entropy_converged"
                        self.log(
                            "\n  [收敛] Stage-1 entropy convergence reached: "
                            f"entropy={entropy:.4f} < threshold={self.stage1_entropy_stop_threshold:.4f}; "
                            f"stopping at episode {episode + 1}."
                        )
                        break


                if is_graceful_stop_requested(stage1_stop_flag_path):
                    self.log(
                        f"\n  [优雅停止] 已检测到停止请求，正在保存 Stage-1 checkpoint "
                        f"(episode={episode + 1}) ..."
                    )
                    if step_info_chunk_file[0] is not None:
                        step_info_chunk_file[0].flush()
                    _stage1_structured_jsonl_sizes = (
                        stage1_data_writer.committed_jsonl_sizes()
                    )
                    _stage1_detail_file_sizes = stage1_detail_file_sizes(
                        step_info_details_dir
                    )
                    save_stage1_rl_checkpoint(
                        path=stage1_checkpoint_path,
                        gtrxl_net=gtrxl_net,
                        optimizer=optimizer,
                        episode=episode,
                        gtrxl_ppo_update_count=gtrxl_ppo_update_count,
                        episode_rewards=episode_rewards,
                        episode_losses=episode_losses,
                        episode_metric1s=episode_metric1s,
                        episode_metric2s=episode_metric2s,
                        episode_entropies=episode_entropies,
                        best_reward=best_reward,
                        best_cost=best_cost,
                        best_config=best_config,
                        search_best_config=search_best_config,
                        global_best_config=global_best_config,
                        window_best_reward=window_best_reward,
                        window_best_cost=window_best_cost,
                        window_best_config=window_best_config,
                        ev_runtime_state={
                            "reward_history": list(self.reward_history),
                            "reward_mean": float(self.reward_mean),
                            "reward_std": float(self.reward_std),
                            "current_episode": int(self.current_episode),
                            "return_normalizer_mean": float(self.return_normalizer.mean),
                            "return_normalizer_var": float(self.return_normalizer.var),
                            "return_normalizer_count": float(self.return_normalizer.count),
                        },
                        stage1_prev_avg_reward=stage1_prev_avg_reward[0],
                        stage1_warnings=stage1_warnings,
                        dataset_protocol_hash=self.dataset_protocol_hash,
                        structured_run_id=stage1_data_writer.run_id,
                        structured_jsonl_sizes=_stage1_structured_jsonl_sizes,
                        detail_file_sizes=_stage1_detail_file_sizes,
                        cuda_rng_role_registry=_stage1_cuda_rng_role_registry,
                    )
                    consume_stop_flag_file(stage1_stop_flag_path)
                    stage1_data_writer.write_summary({
                        "status": "stopped",
                        "stop_reason": "graceful_stop",
                        "completed_episodes": int(episode + 1),
                        "best_reward": float(best_reward),
                        "best_cost": float(best_cost),
                        "best_config": best_config,
                    })
                    stage1_data_writer.close()
                    if self.run_output_dir:
                        update_persistent_metadata_stage(
                            self.run_output_dir, "stage1_search", "in_progress",
                            extra_fields={
                                "completed_episodes": episode + 1,
                                "total_episodes": (
                                    None
                                    if self.stage1_rl_unbounded_until_entropy
                                    else int(self.stage1_rl_episode_limit)
                                ),
                                "stopped_by": "graceful_stop",
                            },
                        )
                    self.log(
                        f"  [优雅停止] checkpoint 已写入 → {stage1_checkpoint_path}\n"
                        f"  下次用相同参数直接运行即可自动续训练（rl/ga 持久化目录），"
                        f"或使用 --resume-from 指向本 run 目录（general-rl）。"
                    )
                    uninstall_graceful_stop_handler()
                    raise SystemExit(0)

            if step_info_chunk_file[0] is not None:
                step_info_chunk_file[0].close()
                step_info_chunk_file[0] = None

            if stage1_warnings:
                _write_warning_report(stage1_warning_file, stage1_warnings, stage_label="阶段1（Stage-1 RL）")
                self.log(f"  ⚠ 共检测到 {len(stage1_warnings)} 次奖励骤降警告，详见: {stage1_warning_file}")


            _s1_final_ep = max(stage1_completed_episodes - 1, stage1_resume_start_episode - 1)
            if stage1_completed_episodes > stage1_resume_start_episode:
                _stage1_structured_jsonl_sizes = (
                    stage1_data_writer.committed_jsonl_sizes()
                )
                _stage1_detail_file_sizes = stage1_detail_file_sizes(
                    step_info_details_dir
                )
                save_stage1_rl_checkpoint(
                    path=stage1_checkpoint_path,
                    gtrxl_net=gtrxl_net,
                    optimizer=optimizer,
                    episode=_s1_final_ep,
                    gtrxl_ppo_update_count=gtrxl_ppo_update_count,
                    episode_rewards=episode_rewards,
                    episode_losses=episode_losses,
                    episode_metric1s=episode_metric1s,
                    episode_metric2s=episode_metric2s,
                    episode_entropies=episode_entropies,
                    best_reward=best_reward,
                    best_cost=best_cost,
                    best_config=best_config,
                    search_best_config=search_best_config,
                    global_best_config=global_best_config,
                    window_best_reward=window_best_reward,
                    window_best_cost=window_best_cost,
                    window_best_config=window_best_config,
                    ev_runtime_state={
                        "reward_history": list(self.reward_history),
                        "reward_mean": float(self.reward_mean),
                        "reward_std": float(self.reward_std),
                        "current_episode": int(self.current_episode),
                        "return_normalizer_mean": float(self.return_normalizer.mean),
                        "return_normalizer_var": float(self.return_normalizer.var),
                        "return_normalizer_count": float(self.return_normalizer.count),
                    },
                    stage1_prev_avg_reward=stage1_prev_avg_reward[0],
                    stage1_warnings=stage1_warnings,
                    dataset_protocol_hash=self.dataset_protocol_hash,
                    structured_run_id=stage1_data_writer.run_id,
                    structured_jsonl_sizes=_stage1_structured_jsonl_sizes,
                    detail_file_sizes=_stage1_detail_file_sizes,
                    cuda_rng_role_registry=_stage1_cuda_rng_role_registry,
                )
                self.log(f"  [完成] Stage-1 最终 checkpoint 已保存 → {stage1_checkpoint_path}")
            if self.run_output_dir:
                update_persistent_metadata_stage(
                    self.run_output_dir, "stage1_search", "completed",
                    extra_fields={
                        "episodes": int(stage1_completed_episodes),
                        "completed_episodes": int(stage1_completed_episodes),
                        "target_episodes": (
                            None
                            if self.stage1_rl_unbounded_until_entropy
                            else int(self.stage1_rl_episode_limit)
                        ),
                        "stop_reason": stage1_stop_reason,
                        "best_reward": float(best_reward),
                        "best_cost": float(best_cost),
                    },
                )
            stage1_data_writer.write_summary({
                "status": "completed",
                "stop_reason": stage1_stop_reason,
                "completed_episodes": int(stage1_completed_episodes),
                "target_episodes": (
                    None
                    if self.stage1_rl_unbounded_until_entropy
                    else int(self.stage1_rl_episode_limit)
                ),
                "ppo_updates": int(gtrxl_ppo_update_count),
                "best_reward": float(best_reward),
                "best_cost": float(best_cost),
                "best_config": best_config,
                "warnings": stage1_warnings,
                "episode_count": len(episode_rewards),
                "entropy_points": len(episode_entropies),
            })
            stage1_data_writer.close()


            try:
                _portable_path = os.path.join(
                    os.path.dirname(self.stage1_step_info_file),
                    "stage1_policy.pt",
                )
                torch.save(
                    {
                        "version": 1,
                        "kind": "stage1_gtrxl_policy",
                        "net_state_dict": gtrxl_net.state_dict(),
                        "arch": {
                            "num_layers": int(self.total_layers),
                            "d_model": int(GTRXL_D_MODEL),
                            "n_heads": int(GTRXL_N_HEADS),
                            "n_gtrxl_layers": int(GTRXL_N_LAYERS),
                            "d_ff": int(GTRXL_D_FF),
                            "dropout": float(GTRXL_DROPOUT),
                        },
                        "metadata": {
                            "trained_episodes": int(stage1_completed_episodes),
                            "best_reward": float(best_reward),
                            "best_cost": float(best_cost),
                            "error_threshold": float(self.error_threshold),
                            "correlation_drop_ratio": float(self.correlation_drop_ratio),
                        },
                    },
                    _portable_path,
                )
                self.log(f"  Stage-1 policy saved to {_portable_path}")
            except Exception as _e:
                self.log(f"  [迁移][警告] portable policy 保存失败：{_e}")


            try:
                _diag = detect_rl_local_optimum(
                    episode_returns=episode_rewards,
                    episode_entropies=episode_entropies,
                    best_score_history=None,
                    action_history=None,
                    window=max(50, int(max(1, stage1_completed_episodes) * 0.1)),
                )
                _report_path = os.path.join(
                    os.path.dirname(self.stage1_step_info_file),
                    "pruning_search_log.txt",
                )
                with open(_report_path, "w", encoding="utf-8") as _f:
                    _f.write("=== Stage-1 RL 局部最优检测报告 ===\n")
                    _f.write(f"完成回合数: {len(episode_rewards)}\n")
                    _f.write(f"判定: {_diag['summary']}\n\n")
                    _f.write("--- 各项判据信号 ---\n")
                    for k, v in _diag["signals"].items():
                        _f.write(f"  {k}: {v}\n")
                    _f.write("\n--- 数值指标 ---\n")
                    for k, v in _diag["metrics"].items():
                        _f.write(f"  {k}: {v}\n")
                    _f.write("\n--- 说明 ---\n")
                    _f.write(
                        "判定规则：A.熵塌缩 / B.reward 平台 / C.best 长期不更新 三条中\n"
                        "≥2 条成立 → likely_local_optimum=True；或 D.动作分布塌缩单独成立。\n"
                    )
                self.log(f"  [检测] 局部最优检测报告 → {_report_path}")
                self.log(f"  [检测] {_diag['summary']}")
            except Exception as _e:
                self.log(f"  [检测][警告] 局部最优检测失败：{_e}")

            best_config, used_baseline = self._select_stage1_reward_best_config(
                best_config,
                best_reward,
                base_gelu,
                base_softmax,
                base_tot_c,
            )

            if used_baseline:
                self.log("\n未找到可行解，使用基线配置。")
            else:
                self.log(
                    "\n[Stage-1] 最终配置使用原始 PPO reward-best；"
                    "GELU/Softmax 为确定性替换，不再按 window 二次评估重排。"
                )

            self.log(f"\n--- PPO训练完成（PPO Training Completed） ---")
            self.log(f"已找到最优配置（通过RL）（Best Configuration Found by RL）：")
            self.log(f"  GELU配置: {best_config['gelu'].tolist()}")
            self.log(f"  Softmax配置: {best_config['softmax'].tolist()}")
            self.log(f"  成本（Cost）: {best_config['cost']:.2f}, 奖励（Reward）: {best_config['reward']:.4f}")


        if len(episode_rewards) > 0:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                episodes = np.arange(1, len(episode_rewards) + 1)
                rewards = np.array(episode_rewards, dtype=np.float32)
                losses = np.array(episode_losses, dtype=np.float32)
                metric1s = np.array(episode_metric1s, dtype=np.float32)
                metric2s = np.array(episode_metric2s, dtype=np.float32)


                metric_names_tuple = self.get_metric_names()
                _num_m = self.get_num_metrics()
                _m1_name = metric_names_tuple[0]
                _m2_name = metric_names_tuple[1] if _num_m > 1 else None


                window = min(max(5, PPO_UPDATE_INTERVAL // 5), 50)

                def compute_ma(data):
                    if len(data) >= window:
                        kernel = np.ones(window, dtype=np.float32) / window
                        return np.convolve(data, kernel, mode="valid")
                    return data

                rewards_ma = compute_ma(rewards)
                losses_ma = compute_ma(losses)
                metric1s_ma = compute_ma(metric1s)
                metric2s_ma = compute_ma(metric2s) if _num_m > 1 else None

                if len(rewards) >= window:
                    episodes_ma = episodes[window - 1:]
                else:
                    episodes_ma = episodes

                dataset_info = f" ({self.data_path})"
                val_guided_info = " [Train Probe]"

                if _num_m == 1:
                    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                    fig.suptitle(f"PPO Training Curves{dataset_info}{val_guided_info}", fontsize=14, fontweight='bold')

                    ax1 = axes[0]
                    ax1.plot(episodes, rewards, label="Episode Reward", alpha=0.6, color='blue')
                    ax1.plot(episodes_ma, rewards_ma, label=f"Moving Avg ({window})", linewidth=2, color='darkblue')
                    ax1.set_xlabel("Episode"); ax1.set_ylabel("Reward")
                    ax1.set_title("Episode Reward"); ax1.grid(True, alpha=0.3); ax1.legend()

                    ax2 = axes[1]
                    ax2.plot(episodes, losses, label="Loss", alpha=0.6, color='red')
                    ax2.plot(episodes_ma, losses_ma, label=f"Moving Avg ({window})", linewidth=2, color='darkred')
                    ax2.set_xlabel("Episode"); ax2.set_ylabel("Loss")
                    ax2.set_title("Loss (lower is better)"); ax2.grid(True, alpha=0.3)
                    ax2.axhline(y=base_loss, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Baseline')
                    ax2.legend()

                    ax3 = axes[2]
                    ax3.plot(episodes, metric1s, label=_m1_name, alpha=0.6, color='green')
                    ax3.plot(episodes_ma, metric1s_ma, label=f"Moving Avg ({window})", linewidth=2, color='darkgreen')
                    ax3.set_xlabel("Episode"); ax3.set_ylabel(_m1_name)
                    ax3.set_title(f"{_m1_name} (higher is better)"); ax3.grid(True, alpha=0.3)
                    ax3.axhline(y=base_p, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Baseline')
                    ax3.legend()
                else:
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                    fig.suptitle(f"PPO Training Curves{dataset_info}{val_guided_info}", fontsize=14, fontweight='bold')

                    ax1 = axes[0, 0]
                    ax1.plot(episodes, rewards, label="Episode Reward", alpha=0.6, color='blue')
                    ax1.plot(episodes_ma, rewards_ma, label=f"Moving Avg ({window})", linewidth=2, color='darkblue')
                    ax1.set_xlabel("Episode"); ax1.set_ylabel("Reward")
                    ax1.set_title("Episode Reward"); ax1.grid(True, alpha=0.3); ax1.legend()

                    ax2 = axes[0, 1]
                    ax2.plot(episodes, losses, label="Loss", alpha=0.6, color='red')
                    ax2.plot(episodes_ma, losses_ma, label=f"Moving Avg ({window})", linewidth=2, color='darkred')
                    ax2.set_xlabel("Episode"); ax2.set_ylabel("Loss")
                    ax2.set_title("Loss (lower is better)"); ax2.grid(True, alpha=0.3)
                    ax2.axhline(y=base_loss, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Baseline')
                    ax2.legend()

                    ax3 = axes[1, 0]
                    ax3.plot(episodes, metric1s, label=_m1_name, alpha=0.6, color='green')
                    ax3.plot(episodes_ma, metric1s_ma, label=f"Moving Avg ({window})", linewidth=2, color='darkgreen')
                    ax3.set_xlabel("Episode"); ax3.set_ylabel(_m1_name)
                    ax3.set_title(f"{_m1_name} (higher is better)"); ax3.grid(True, alpha=0.3)
                    ax3.axhline(y=base_p, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Baseline')
                    ax3.legend()

                    ax4 = axes[1, 1]
                    ax4.plot(episodes, metric2s, label=_m2_name, alpha=0.6, color='purple')
                    ax4.plot(episodes_ma, metric2s_ma, label=f"Moving Avg ({window})", linewidth=2, color='darkviolet')
                    ax4.set_xlabel("Episode"); ax4.set_ylabel(_m2_name)
                    ax4.set_title(f"{_m2_name} (higher is better)"); ax4.grid(True, alpha=0.3)
                    ax4.axhline(y=base_s, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Baseline')
                    ax4.legend()

                plot_path = self.stage1_training_curve_path
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150)
                plt.close()
                self.log(f"PPO训练曲线已保存至（PPO training curves saved to）: {plot_path}")

                if episode_entropies:
                    update_episodes = np.arange(PPO_UPDATE_INTERVAL, len(episode_rewards) + 1, PPO_UPDATE_INTERVAL)
                    entropies = np.array(episode_entropies, dtype=np.float32)
                    if len(update_episodes) == len(entropies):
                        fig_ent, ax_ent = plt.subplots(1, 1, figsize=(10, 5))
                        ax_ent.plot(update_episodes, entropies, label="Policy Entropy", alpha=0.8, color='teal', marker='o', markersize=3)
                        window_ent = min(5, max(1, len(entropies) // 5))
                        if len(entropies) >= window_ent:
                            kernel_ent = np.ones(window_ent, dtype=np.float32) / window_ent
                            ent_ma = np.convolve(entropies, kernel_ent, mode="valid")
                            ax_ent.plot(update_episodes[window_ent - 1:], ent_ma, label=f"Moving Avg ({window_ent})", linewidth=2, color='darkgreen')
                        ax_ent.set_xlabel("Episode (at PPO update)")
                        ax_ent.set_ylabel("Entropy")
                        ax_ent.set_title("PPO Training: Policy Entropy over Episodes")
                        ax_ent.grid(True, alpha=0.3)
                        ax_ent.legend()
                        plt.tight_layout()
                        entropy_plot_path = self.stage1_entropy_curve_path
                        plt.savefig(entropy_plot_path, dpi=150)
                        plt.close()
                        self.log(f"PPO熵曲线已保存至（PPO entropy curve saved to）: {entropy_plot_path}")
                    else:
                        self.log(f"[警告] 熵曲线未绘制（Entropy curve not plotted）: update_episodes长度={len(update_episodes)}, entropies长度={len(entropies)}")
            except Exception as e:
                self.log(f"[警告] PPO训练曲线绘制失败（Failed to plot PPO training curves）: {e}")


        if (
                getattr(self, "decoupled_layout", False)
                and self.skip_noise_rl
                and not self.skip_stage1_rl
                and self.blb_v3_search_backend == "ppo"
        ):
            self._maybe_snapshot_decoupled_stage1_record(
                best_config=best_config,
                base_gelu=base_gelu,
                base_softmax=base_softmax,
                episode_metric1s=episode_metric1s,
                episode_metric2s=episode_metric2s,
                episode_losses=episode_losses,
                best_reward=best_reward,
                best_cost=best_cost,
                completed_episodes=stage1_completed_episodes,
            )


        self.log("\n" + "="*60)
        self.log("最终评估报告（FINAL EVALUATION REPORT）（验证集）")
        self.log("="*60)


        noise_stage_result = None
        final_eval_result = None
        final_eval_error = None
        final_eval_ineligible_reason = None
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
            ) = self._resolve_stage2_fixed_stage1_config(search_best_config=best_config)
            if (
                    self.blb_v3_search_backend != "ppo"
                    and best_config is not None
            ):
                stage2_fixed_source = (
                    f"stage1_{self.blb_v3_search_backend}_result"
                )
                stage2_fixed_label = (
                    f"Stage-1 {self.blb_v3_search_backend} selected config"
                )
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


            if self.skip_final_eval:
                self.log("\n[信息] 统一最终评估已跳过（--skip-final-eval）。")
                if self.run_output_dir:
                    update_persistent_metadata_stage(
                        self.run_output_dir, "final_eval", "skipped")
            else:


                prior_stage1_search_best = None
                prior_stage2_search_best = None
                if self.final_eval_only:
                    prior_stage1_search_best, prior_stage2_search_best = (
                        self._load_prior_rl_search_results()
                    )

                stage1_search_best = None
                if best_config is not None:
                    stage1_search_best = {
                        "gelu": np.asarray(best_config["gelu"], dtype=int),
                        "softmax": np.asarray(best_config["softmax"], dtype=int),
                    }
                elif prior_stage1_search_best is not None:
                    stage1_search_best = prior_stage1_search_best

                stage2_search_best = None
                noise_limit_loss = limit_loss
                noise_limit_p = limit_p
                noise_limit_s = limit_s
                if noise_stage_result is not None:
                    final_eval_ineligible_reason = (
                        self._stage2_final_eval_ineligible_reason(
                            noise_stage_result,
                        )
                    )
                    if final_eval_ineligible_reason is None:
                        stage2_search_best = (
                            self._build_stage2_final_eval_handoff(
                                noise_stage_result,
                            )
                        )
                    noise_limit_loss = noise_stage_result.get("limit_loss", limit_loss)
                    noise_limit_p = noise_stage_result.get("limit_p", limit_p)
                    noise_limit_s = noise_stage_result.get("limit_s", limit_s)
                    baseline_noise_tot_c = noise_stage_result.get("baseline_tot_c")
                else:
                    baseline_noise_cfg = self._get_max_noise_configuration()
                    baseline_noise_tot_c, _ = self.get_noise_simulated_cost(**baseline_noise_cfg)
                    if stage2_search_best is None and prior_stage2_search_best is not None:
                        stage2_search_best = prior_stage2_search_best

                if final_eval_ineligible_reason is not None:
                    self.log(
                        "[Comparator][Paean] skipping optional final-eval: "
                        + final_eval_ineligible_reason
                    )
                    final_eval_result = None
                elif self.final_eval_only:
                    final_eval_result = self.run_unified_final_eval(
                        stage1_search_best=stage1_search_best,
                        stage2_search_best=stage2_search_best,
                        baseline_stage1_gelu=base_gelu,
                        baseline_stage1_softmax=base_softmax,
                        baseline_noise_tot_c=baseline_noise_tot_c,
                        limit_loss=noise_limit_loss,
                        limit_p=noise_limit_p,
                        limit_s=noise_limit_s,
                    )
                elif getattr(self, "decoupled_layout", False):


                    self.log(
                        "[解耦] 跳过训练末自动 final-eval；如需重型同-cost 对比，"
                        "请用独立 final-eval 工具（--stage stage1|stage2 --record-dir ...）。"
                    )
                    final_eval_result = None
                else:
                    from Paean.embedded import run_embedded_final_eval

                    try:
                        final_eval_result = run_embedded_final_eval(
                            evaluator=self,
                            search_best_stage1=stage1_search_best,
                            search_best_stage2=stage2_search_best,
                            baseline_stage1_gelu=base_gelu,
                            baseline_stage1_softmax=base_softmax,
                            baseline_noise_tot_c=baseline_noise_tot_c,
                            limit_loss=noise_limit_loss,
                            limit_p=noise_limit_p,
                            limit_s=noise_limit_s,
                            preset_name=self.final_eval_preset,
                            output_root=self.final_eval_output_root,
                            run_name=self.final_eval_run_name,
                        )
                    except Exception as exc:
                        if self.blb_v3_search_backend == "ppo":
                            raise
                        final_eval_error = exc
                        self.log(
                            "[Comparator][Paean] optional final-eval failed "
                            "after strict F4; preserving the "
                            f"two-stage search artifact: {exc!r}"
                        )
                if self.run_output_dir:
                    update_persistent_metadata_stage(
                        self.run_output_dir,
                        "final_eval",
                        (
                            "skipped_ineligible"
                            if final_eval_ineligible_reason is not None
                            else (
                                "failed_optional"
                                if final_eval_error is not None else "completed"
                            )
                        ),
                    )
        finally:
            self.active_log_file = previous_log_file

        self.last_noise_stage_result = noise_stage_result
        self.last_final_eval_result = final_eval_result
        ordinary_two_stage_payload = None

        if (
                self.blb_v3_search_backend != "ppo"
                and best_config is not None
                and noise_stage_result is not None
                and self.run_output_dir
        ):
            if final_eval_result is not None:
                final_eval_status = "completed"
            elif final_eval_ineligible_reason is not None:
                final_eval_status = "skipped_ineligible"
            elif final_eval_error is not None:
                final_eval_status = "failed_optional"
            elif self.skip_final_eval:
                final_eval_status = "skipped_by_request"
            else:
                final_eval_status = "decoupled_not_run"

            ordinary_two_stage_payload = _build_ordinary_two_stage_result(
                backend=self.blb_v3_search_backend,
                stage1_best_config=best_config,
                stage2_result=noise_stage_result,
                final_eval_result=final_eval_result,
                final_eval_status=final_eval_status,
                final_eval_ineligible_reason=(
                    final_eval_ineligible_reason
                ),
                final_eval_error=(
                    None if final_eval_error is None else repr(final_eval_error)
                ),
            )
            from blb_stage2_rl.search_baseline_runner import _atomic_json

            _atomic_json(
                os.path.join(self.run_output_dir, "two_stage_result.json"),
                ordinary_two_stage_payload,
            )

        if ordinary_two_stage_payload is not None:
            opt_gelu = np.asarray(
                ordinary_two_stage_payload["stage1"]["gelu_degrees"],
                dtype=int,
            )
            opt_softmax = np.asarray(
                ordinary_two_stage_payload["stage1"][
                    "softmax_degrees"
                ],
                dtype=int,
            )
        elif final_eval_result is not None:
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
