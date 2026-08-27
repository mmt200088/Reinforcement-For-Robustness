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
)
from rfr.preparation.data.protocol import TRAIN_PROBE_SPLIT, validate_dataset
from rfr.preparation.data.mrpc_contract import (
    MRPC_STAGE2_RL_ALIGNMENT_BATCH_SIZE,
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


NOISE_STAGE_STEP_INFO_FILE = "noise_ppo_step_info.txt"
NOISE_STAGE_TRAINING_CURVE_PATH = "noise_ppo_training_curve.png"
NOISE_STAGE_ENTROPY_CURVE_PATH = "noise_ppo_entropy_curve.png"
DEFAULT_STAGE1_SEARCH_LOG_FILE = "search.log"

SEARCH_LOG_HEADER = "=== PPO强化学习优化日志已启动（PPO RL Optimization Log Started） ==="
DEFAULT_STAGE1_STEP_INFO_FILE = "ppo_step_info.txt"
DEFAULT_STAGE1_TRAINING_CURVE_FILE = "ppo_training_curve.png"
DEFAULT_STAGE1_ENTROPY_CURVE_FILE = "ppo_entropy_curve.png"
DEFAULT_FINAL_EVAL_DIR = os.path.join("outputs", "rl", "evaluation")
DEFAULT_NOISE_PROGRESS_DIR = os.path.join("outputs", "rl", "stage2", "progress")


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
    """Update stage-completion fields in a persistent run metadata file."""
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
        stage2_noise_dir = os.path.join(run_output_dir, "stage2")
    stage2_noise_progress_dir = os.path.join(stage2_noise_dir, "progress")
    final_eval_dir = os.path.join(run_output_dir, "evaluation")

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
        """Normalize values with the running mean and variance."""
        if isinstance(x, torch.Tensor):
            return (x - self.mean) / (self.std + 1e-8)
        return (x - self.mean) / (self.std + 1e-8)

    def denormalize(self, x):
        """Restore values from running normalization."""
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
        """Start recording a recurrent rollout episode."""
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
        """Append one Stage 1 decision to the current episode."""
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
        """Finalize and store the current episode."""
        self.episodes.append(self._current)
        self._current = None

    def clear(self):
        """Discard all stored rollout episodes."""
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
        """Reset the Stage 1 search state."""
        self.current_layer = 0
        self.accumulated_cost = 0.0
        self.gelu_config = []
        self.softmax_config = []
        self.prev_gelu_degree = 4


        self.accumulated_dense_reward = 0.0


        self.gelu_history = np.full(self.total_layers, HISTORY_MASK_VALUE, dtype=np.float32)


        return self._get_state()

    def get_gelu_action_mask(self, layer_idx=None):
        """Return the current GELU action mask without mutating the environment."""
        del layer_idx
        return STAGE1_GELU_ACTION_MASK.copy()

    def _get_state(self):
        """Build the state vector from selected degrees and the current layer."""


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
        """Compute a progress reward from the selected approximation degrees."""
        step_cost = GELU_COST[gelu_degree] + SOFTMAX_COST[FIXED_SOFTMAX_DEGREE]


        cost_saving = (self.max_cost_per_layer - step_cost) / self.max_cost_per_layer
        cost_reward = REWARD_DENSE_SCALE * cost_saving
        return cost_reward

    def step(self, gelu_action_idx):
        """Apply one layer decision and return the next transition."""

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
            """Apply a smooth log barrier to a constraint margin."""
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
        self.current_entropy_coef = PPO_ENTROPY_INITIAL
        self.current_lr = self.stage1_ppo_lr_initial


        self.reward_history = deque(maxlen=RUNNING_REWARD_HISTORY_SIZE)
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
                or self.stage1_rl_episodes != PPO_MAX_EPISODES
        ):
            raise ValueError(
                "stage1_rl_episodes was set while Stage 1 is skipped"
            )
        if (
                not self.skip_stage1_rl
                and self.stage1_rl_episodes < PPO_UPDATE_INTERVAL
        ):
            raise ValueError(
                f"stage1_rl_episodes must be at least {PPO_UPDATE_INTERVAL}"
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
        """Resolve the checkpoint path used for Stage 1 resume."""
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


    def compute_gae(self, rewards, values, dones, gamma=PPO_GAMMA, lam=PPO_LAMBDA):
        """Compute generalized advantage estimates for one rollout."""
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
            int(getattr(self, "random_seed", 42))
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
        from rfr.search.rl.stage1.parallel_runner import build_stage1_parallel_runner
        from rfr.search.runtime.device_utils import parse_device_ids

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
            from rfr.common.json_utils import read_json_file
            from rfr.search.comparators.common.stage1_core import (
                Stage1Constraints,
                stage1_comparator_search_config,
                validate_stage1_comparator_setup,
            )
            from rfr.search.comparators.common.stage1_runner import (
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
            stage1_search_config = stage1_comparator_search_config(
                backend,
                bo_no_improvement_patience=(
                    self.comparator_bo_stage1_no_improvement
                ),
                greedy_no_improvement_rounds=(
                    self.comparator_greedy_stage1_no_improvement_rounds
                ),
                ga_update_generations=(
                    self.comparator_ga_stage1_generations
                ),
            )
            validate_stage1_comparator_setup(
                backend=backend,
                config=stage1_search_config,
                num_layers=int(self.total_layers),
                constraints=stage1_constraints,
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
                    backend == "greedy"
                    and stage1_comparator_result.termination_reason
                    != "consecutive_no_improvement_rounds"
            ):
                raise RuntimeError(
                    "Stage-1 Greedy did not verify every 1-opt and 2-opt "
                    "local optimum"
                )
            if (
                    backend == "coinn_ga"
                    and (
                        stage1_comparator_result.termination_reason
                        != "completed_generations"
                        or int(
                            stage1_comparator_result.config.ga_update_generations
                        ) != self.comparator_ga_stage1_generations
                    )
            ):
                raise RuntimeError(
                    "Stage-1 COINN-GA did not satisfy its configured full-generation "
                    "contract"
                )

            stage1_result_path = os.path.join(
                stage1_output_dir, "result.json",
            )
            with open(stage1_result_path, "rb") as result_handle:
                stage1_result_sha256 = hashlib.sha256(
                    result_handle.read()
                ).hexdigest()
            stage1_manifest = read_json_file(
                os.path.join(stage1_output_dir, "manifest.json")
            )
            if not isinstance(stage1_manifest, Mapping):
                raise RuntimeError("Stage-1 manifest must be a JSON object")
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
                "result_sha256": stage1_result_sha256,
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
                root_dir=os.path.join(
                    os.path.dirname(self.stage1_step_info_file),
                    "records",
                ),
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
                "stage1_episodes_requested": int(self.stage1_rl_episode_limit),
                "termination": "maximum_episodes",
                "ppo_update_interval": int(PPO_UPDATE_INTERVAL),
                "ppo_lr_initial": float(self.stage1_ppo_lr_initial),
                "stage1_rl_devices": self.stage1_rl_devices,
                "random_seed": int(getattr(self, "random_seed", 42)),
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
                self.log(
                    f"  已恢复至回合 {stage1_resume_start_episode}，"
                    f"将从回合 {stage1_resume_start_episode + 1} "
                    f"继续训练至 {self.stage1_rl_episodes}"
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
            _stage1_episode_iter = range(
                stage1_resume_start_episode, self.stage1_rl_episode_limit,
            )
            for episode in _stage1_episode_iter:

                current_lr, current_entropy = self.update_hyperparameters(optimizer, episode)


                _handled_via_parallel = False
                if _stage1_parallel_runner is not None:
                    if not _stage1_parallel_stash:
                        _window_idx_for_runner = episode // PPO_UPDATE_INTERVAL
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
                            base_seed=int(getattr(self, "random_seed", 42)),
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
                        _s1_remain = self.stage1_rl_episode_limit - (episode + 1)
                        _s1_eta = _s1_avg_ep * _s1_remain
                        _s1_progress_title = (
                            f"Stage-1 RL 进度 · 回合 {episode + 1} / "
                            f"{self.stage1_rl_episode_limit}"
                        )
                        _s1_progress_lines = [
                            _progress_bar(episode + 1, self.stage1_rl_episode_limit),
                            *_s1_best_lines,
                            f"已用时: {_fmt_elapsed(_s1_elapsed)}  "
                            f"预计剩余: {_fmt_elapsed(_s1_eta)}  "
                            f"预计完成: {_fmt_eta_finish(_s1_eta)}  "
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

                    if (episode + 1) < self.stage1_rl_episode_limit:
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
                    if not _stage1_reached_episode_cap:
                        if _stage1_parallel_runner is not None:
                            _deferred_gpu_failure = (
                                _stage1_parallel_runner.pop_deferred_gpu_failure()
                            )
                            if _deferred_gpu_failure is not None:
                                raise _deferred_gpu_failure
                        raise_if_elastic_gpu_restart_requested()
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
                                "total_episodes": int(self.stage1_rl_episode_limit),
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
                        "target_episodes": int(self.stage1_rl_episode_limit),
                        "stop_reason": stage1_stop_reason,
                        "best_reward": float(best_reward),
                        "best_cost": float(best_cost),
                    },
                )
            stage1_data_writer.write_summary({
                "status": "completed",
                "stop_reason": stage1_stop_reason,
                "completed_episodes": int(stage1_completed_episodes),
                "target_episodes": int(self.stage1_rl_episode_limit),
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


        if best_config is not None and not self.skip_stage1_rl:
            self._write_stage1_best_config_file(
                best_config=best_config,
                completed_episodes=stage1_completed_episodes,
            )

        self.log("\n" + "="*60)
        self.log("最终评估报告（FINAL EVALUATION REPORT）（验证集）")
        self.log("="*60)


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
