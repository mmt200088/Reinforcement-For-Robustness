"""BLB Stage 2 RL ``Env`` 包装。

不强依赖 ``gymnasium`` —— 我们只用最小化的 ``reset/step`` 接口，避免新增依赖。
"""
from __future__ import annotations

import copy
import contextlib
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

# Reusable no-op context for the single-worker-per-device path.
_NULL_CTX = contextlib.nullcontext()

from blb_rl_bridge import BLBNoiseRLBridge
from rescale_optimizer_bridge import (
    RescaleOptimizerBridge,
    _strip_layer_suffix,
    aggregate_optimizer_signals,
    apply_optimizer_output_to_cfg,
    apply_rotation_flags_to_cfg,
    sync_block2_aux_fresh_binding,
    sync_block2_qk_binding,
    sync_block4_v_mask_binding,
    sync_block5_aux_fresh_binding,
)

from .action_space import (
    ActionDecodeResult,
    BLB_FIRST_INPUT_N,
    K_LEVELS,
    MaxSFsTable,
    action_dims_for_config,
    action_vector_to_cfgs,
    avg_truncation_k_in_action,
    build_optimizer_requests,
    layer_dims,
    make_all_max_action_vector,
    parse_config_name,
)
from .candidate_store import action_hash
from .optimizer_cost import evaluate_action_for_cost
from .probe_runner import ProbeRunner, format_diagnostics_line
from .reward import (
    BaselineCostStats,
    EpisodeMetrics,
    RewardBreakdown,
    RewardWeights,
    compute_reward,
)


# ---------------------------------------------------------------------------
# 评估子集 / forward 钩子（薄薄一层，绕过 evaluator 的 attention noise 输入参数）
# ---------------------------------------------------------------------------
@dataclass
class ProbeBatch:
    """一个评估 mini-batch 的 (input_ids, attention_mask, labels, token_type_ids?)。"""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    token_type_ids: Optional[torch.Tensor] = None

    @classmethod
    def from_batch(cls, batch: Mapping[str, torch.Tensor], device: torch.device) -> "ProbeBatch":
        ii = batch["input_ids"].to(device, non_blocking=True)
        am = batch.get("attention_mask")
        am = am.to(device, non_blocking=True) if am is not None else torch.ones_like(ii)
        lb = batch["labels"].to(device, non_blocking=True)
        tt = batch.get("token_type_ids")
        tt = tt.to(device, non_blocking=True) if tt is not None else None
        return cls(input_ids=ii, attention_mask=am, labels=lb, token_type_ids=tt)


def _logits_to_classes(preds: torch.Tensor) -> torch.Tensor:
    if preds.dim() == 1:
        return (preds > 0.5).long()
    return preds.argmax(dim=-1)


def _compute_metrics_on_batch(
        logits: torch.Tensor,
        labels: torch.Tensor,
        is_regression: bool = False,
        ) -> Tuple[float, float, float]:
    """返回 (loss, metric1, metric2)。

    跑 cross-entropy loss + 简单 accuracy；用于 RL 评估子集（不追求与最终 final-eval
    完全相同的指标，只要差分单调即可指导 RL）。
    """
    if labels.dtype == torch.long and not is_regression:
        loss = torch.nn.functional.cross_entropy(
            logits.float(), labels.long(), reduction="mean"
        ).item()
        preds = _logits_to_classes(logits.detach())
        acc = (preds.detach().long() == labels.detach().long()).float().mean().item()
        return float(loss), float(acc), float(acc)
    # 回归 / STSB
    pred_flat = logits.float().reshape(-1)
    label_flat = labels.float().reshape(-1)
    loss = torch.nn.functional.mse_loss(pred_flat, label_flat).item()
    return float(loss), -float(loss), -float(loss)   # metric=-mse 越大越好


# ---------------------------------------------------------------------------
# Env 主类
# ---------------------------------------------------------------------------
def summarize_optimizer_invalid_outputs(
        outputs: Mapping[str, Any],
        *,
        limit: int = 8,
        ) -> str:
    items: List[str] = []
    for name, out in sorted(outputs.items()):
        if bool(getattr(out, "valid", False)):
            continue
        raw = getattr(out, "raw", {}) or {}
        message = ""
        if isinstance(raw, Mapping):
            result = raw.get("result") or {}
            if isinstance(result, Mapping):
                message = str(result.get("message", "") or "")
        invalid_chain = getattr(out, "invalid_chain", None)
        if message:
            detail = message
        elif invalid_chain is not None:
            detail = f"invalid_chain={invalid_chain}"
        else:
            detail = "optimizer output marked invalid"
        items.append(f"{name}: {detail}")
    if not items:
        return ""
    shown = items[:max(1, int(limit))]
    if len(items) > len(shown):
        shown.append(f"... (+{len(items) - len(shown)} more)")
    return "Rescale_optimizer invalid blocks: " + "; ".join(shown)


@dataclass
class BLBStage2EnvConfig:
    """``BLBStage2Env`` 的运行参数。"""
    profile: str = "default"
    num_trials_per_step: int = 3            # spec §5.3 推荐 3 次取 std
    # ADR-012 borderline retest (2026-06-12): a metric fail whose worst
    # per-channel deficit is within reward_weights.near_miss_band gets ONE
    # fresh re-measurement with multiplier x num_trials_per_step trials
    # (salted deterministic probe seed); the retest verdict replaces the
    # first. The 2nd 60k showed ~8% of fusion-era episodes were borderline
    # probe-quantization P1s (m1 a hair under threshold, ZERO catastrophic) —
    # a stochastic -46 hammer that killed all fusion exploration. Only active
    # on the deterministic probe path (probe_noise_seed set).
    borderline_retest_enabled: bool = True
    borderline_retest_trials_multiplier: int = 2
    probe_batch_count: int = 4              # 每次 trial 跑多少 mini-batch
    deterministic_eval: bool = False
    rotation_name_map: Optional[Mapping[Tuple[int, str], Mapping[str, str]]] = None
    persistent_probe_install: bool = False
    """Keep BLB wrappers installed across episodes and update cfgs in-place.

    Multi-GPU reward probes otherwise pay a full clear + reinstall on every
    episode for every model replica. The handler's BLB replace methods are
    idempotent and update existing wrappers/hooks, so persistent mode removes
    the restore/re-wrap churn without changing the action/config semantics.
    """


class BLBStage2Env:
    """把"action → install BLB → forward → metrics → reward"封装成单步 env。

    单步 episode（horizon=1），每次 ``step(action)`` 返回 (state, reward, done=True, info)。

    依赖：
      * ``handler``：``ReversibleLayerHandler`` 实例（已经替换好 GELU/Softmax 近似）
      * ``model``：HF 模型（用于 forward）
      * ``probe_batches``：评估用的 mini-batches list（每条是 ProbeBatch）
      * ``rescale_bridge``：``RescaleOptimizerBridge``（必须 InProcessInvoker；
        训练路径强制 in-process real，初始化失败即中止）
      * ``baseline``：``BaselineCostStats``（训练前算好）
      * ``reward_weights``：``RewardWeights``
      * ``acc_threshold / stab_threshold``：硬约束阈值
      * ``max_sfs``：``MaxSFsTable``（从 JSON 加载或默认）

    Args:
        gelu_degree:    模型每层的 GELU 多项式 degree（block5 用，spec §3.2）
        attn_degree:    模型每层的 softmax 多项式 degree（block3 用）
        layers_attribute: BLB bridge 用的 attribute 路径（含 "model." 前缀）
    """

    def __init__(
            self,
            *,
            handler,
            model,
            probe_batches: Sequence[ProbeBatch],
            rescale_bridge: RescaleOptimizerBridge,
            baseline: BaselineCostStats,
            reward_weights: RewardWeights,
            acc_threshold: float,
            stab_threshold: float,
            max_sfs: MaxSFsTable,
            num_layers: int,
            gelu_degree=4,
            attn_degree=4,
            layers_attribute: str = "model.bert.encoder.layer",
            is_regression: bool = False,
            env_cfg: Optional[BLBStage2EnvConfig] = None,
            probe_runner: Optional[ProbeRunner] = None,
            acc_threshold_m2: Optional[float] = None,
            ):
        self.handler = handler
        self.model = model
        self.probe_batches = list(probe_batches)
        self.rescale_bridge = rescale_bridge
        self.baseline = baseline
        self.reward_weights = reward_weights
        self.acc_threshold = float(acc_threshold)
        # v3 (2026-05-20): second metric (m2) joins the metric_ok gate. When the
        # caller doesn't supply a per-m2 threshold, fall back to the m1
        # threshold (preserves single-metric semantics for legacy code paths).
        self.acc_threshold_m2: Optional[float] = (
            float(acc_threshold_m2) if acc_threshold_m2 is not None else None
        )
        self.stab_threshold = float(stab_threshold)
        self.max_sfs = max_sfs
        self.num_layers = int(num_layers)
        self.gelu_degree = self._normalize_degree_vector(gelu_degree, default=4, name="gelu_degree")
        self.attn_degree = self._normalize_degree_vector(attn_degree, default=4, name="attn_degree")
        self.gelu_degree_state = self._degree_state_scalar(self.gelu_degree, default=4)
        self.attn_degree_state = self._degree_state_scalar(self.attn_degree, default=4)
        self.layers_attribute = str(layers_attribute)
        self.is_regression = bool(is_regression)
        self.env_cfg = env_cfg or BLBStage2EnvConfig()

        self.bridge = BLBNoiseRLBridge(handler, layers_attribute=layers_attribute)
        # Multi-GPU probe parallelism: when set, install/clear/_eval_on_probe
        # route to the runner instead of (self.bridge, self.model). Single-GPU
        # runs leave this None and the existing path runs bitwise-unchanged.
        self.probe_runner = probe_runner
        self._last_probe_diagnostics: Dict[str, Any] = {}
        self._installed_action_hash: Optional[str] = None
        self.pareto_cost_archive = None
        # Counter for derive_probe_base_seed; bumped every action eval so two
        # consecutive actions in the same episode get different seed streams.
        self._probe_eval_counter: int = 0
        # Deterministic probe noise (2026-06-10, episode-parallel path): when
        # set (derived from (run_seed, global_episode) by the runner), the K
        # probe trials run serially on this env's device with the dedicated
        # noise generator reseeded per trial — no wall clock, no global RNG
        # mutation, identical results for any GPU count / trial scheduling.
        # None (default) keeps the legacy true-random behavior bit-for-bit.
        self.probe_noise_seed: Optional[int] = None
        # Per-DEVICE lock shared by same-device episode-parallel workers
        # (workers-per-device > 1, 2026-06-12): each probe trial's
        # (reseed_noise -> full forward) is an RNG-consuming atomic unit on
        # the device's dedicated noise generator. None / uncontended = no-op.
        self.probe_device_lock: Optional[Any] = None

        self.action_dims = action_dims_for_config(self.num_layers)
        self.total_action_dim = len(self.action_dims)

        # state 设计：spec §5.1 minimal state
        self._last_total_bits_norm: float = 0.0
        self._last_fusion_count: float = 0.0
        self._last_invalid_rate: float = 0.0
        self._step_idx: int = 0

        self._device = next(model.parameters()).device

    def clear_installed_blb(self) -> None:
        """Clear BLB noise from the active single- or multi-GPU install path."""
        if self.probe_runner is not None:
            self.probe_runner.clear()
        else:
            self.bridge.clear()
        self._installed_action_hash = None

    def _normalize_degree_vector(self, degrees, *, default: int, name: str):
        if degrees is None:
            return int(default)
        arr = np.asarray(degrees, dtype=int).reshape(-1)
        if arr.size == 0:
            return int(default)
        if arr.size == 1:
            return int(arr[0])
        if arr.size != self.num_layers:
            raise ValueError(f"{name} length {arr.size} must be 1 or num_layers={self.num_layers}")
        return arr.copy()

    @staticmethod
    def _degree_state_scalar(degrees, *, default: int) -> float:
        arr = np.asarray(degrees, dtype=float).reshape(-1)
        if arr.size == 0:
            return float(default)
        return float(arr.mean())

    @staticmethod
    def _attr_path_or_none(root, path: str):
        obj = root
        for part in str(path).split("."):
            if not part:
                continue
            if not hasattr(obj, part):
                return None
            obj = getattr(obj, part)
        return obj

    @staticmethod
    def _degree_values_equal(left, right) -> bool:
        return np.array_equal(
            np.asarray(left, dtype=int).reshape(-1),
            np.asarray(right, dtype=int).reshape(-1),
        )

    def _resolve_transformer_layers(self):
        raw = str(self.layers_attribute or "").strip()
        if not raw:
            return None

        paths: List[str] = [raw]
        if raw.startswith("model."):
            paths.append(raw[len("model."):])
        else:
            paths.append("model." + raw)

        unique_paths: List[str] = []
        for path in paths:
            if path not in unique_paths:
                unique_paths.append(path)

        for root in (self.model, self.handler):
            for path in unique_paths:
                layers = self._attr_path_or_none(root, path)
                if layers is not None:
                    return layers
        return None

    def _read_model_degree_vectors(self) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        layers = self._resolve_transformer_layers()
        if layers is None:
            return None, None
        try:
            layer_list = list(layers)
        except TypeError:
            return None, None
        if len(layer_list) < self.num_layers:
            return None, None

        attn_degrees: List[int] = []
        gelu_degrees: List[int] = []
        for layer in layer_list[:self.num_layers]:
            attn_self = self._attr_path_or_none(layer, "attention.self")
            attn_degree = getattr(attn_self, "degree", None)
            if attn_degree is not None:
                try:
                    attn_degrees.append(int(attn_degree))
                except Exception:
                    attn_degrees = []

            gelu_module = self._attr_path_or_none(layer, "intermediate.intermediate_act_fn")
            gelu_degree = getattr(gelu_module, "degree", None)
            if gelu_degree is not None:
                try:
                    gelu_degrees.append(int(gelu_degree))
                except Exception:
                    gelu_degrees = []

        if len(attn_degrees) != self.num_layers:
            attn_degrees = None
        if len(gelu_degrees) != self.num_layers:
            gelu_degrees = None
        return attn_degrees, gelu_degrees

    def sync_degree_vectors_from_model(self) -> Dict[str, List[int]]:
        """Use the installed polynomial modules as the source of truth for degrees."""
        updates: Dict[str, List[int]] = {}
        try:
            attn_degrees, gelu_degrees = self._read_model_degree_vectors()
        except Exception:
            return updates

        if attn_degrees is not None:
            normalized = self._normalize_degree_vector(attn_degrees, default=4, name="attn_degree")
            if not self._degree_values_equal(self.attn_degree, normalized):
                self.attn_degree = normalized
                self.attn_degree_state = self._degree_state_scalar(self.attn_degree, default=4)
                updates["attn_degree"] = list(attn_degrees)
        if gelu_degrees is not None:
            normalized = self._normalize_degree_vector(gelu_degrees, default=4, name="gelu_degree")
            if not self._degree_values_equal(self.gelu_degree, normalized):
                self.gelu_degree = normalized
                self.gelu_degree_state = self._degree_state_scalar(self.gelu_degree, default=4)
                updates["gelu_degree"] = list(gelu_degrees)
        return updates

    # ------------------------------------------------------------------
    # gym-like 接口
    # ------------------------------------------------------------------
    @property
    def state_dim(self) -> int:
        # 见 _build_state（与设计保持同步）
        return 6 + 4 + self.num_layers     # static + last + per-layer step indicator

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            ) -> np.ndarray:
        """清掉所有 legacy 噪声 + BLB 残留，回到干净状态；返回 obs。"""
        # 1) 防御式清掉旧版 stage2 RL legacy 噪声
        try:
            self.handler.restore_layer_input_noise(layer_indices=list(range(self.num_layers)))
        except Exception:
            pass
        for restore_name in (
                "restore_layer_query_noise", "restore_layer_key_noise", "restore_layer_value_noise",
                "restore_layer_wo_noise", "restore_layer_ffn1_noise", "restore_layer_ffn2_noise",
                "restore_layer_softmax_value_noise"):
            method = getattr(self.handler, restore_name, None)
            if method is None:
                continue
            try:
                method(layer_indices=list(range(self.num_layers)))
            except Exception:
                pass

        # 2) 清掉本 env 之前可能装过的 BLB 噪声（重复 clear 安全）。
        # Multi-GPU persistent mode deliberately keeps wrappers/hooks installed
        # across episodes and only updates cfgs when the next action arrives.
        if not bool(getattr(self.env_cfg, "persistent_probe_install", False)):
            try:
                self.clear_installed_blb()
            except Exception:
                pass

        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed) % (2**32))
            random.seed(int(seed))

        self.sync_degree_vectors_from_model()
        return self._build_state()

    @staticmethod
    def _metrics_from_trial_results(
            results: Sequence[Tuple[float, float, float]],
            ) -> EpisodeMetrics:
        if not results:
            return EpisodeMetrics()
        loss_arr = np.array([float(x[0]) for x in results], dtype=float)
        m1_arr = np.array([float(x[1]) for x in results], dtype=float)
        m2_arr = np.array([float(x[2]) for x in results], dtype=float)
        _LOSS_CAP = 100.0
        loss_arr = np.nan_to_num(loss_arr, nan=_LOSS_CAP, posinf=_LOSS_CAP, neginf=_LOSS_CAP)
        loss_arr = np.clip(loss_arr, 0.0, _LOSS_CAP)
        m1_arr = np.nan_to_num(m1_arr, nan=0.0, posinf=1.0, neginf=0.0)
        m2_arr = np.nan_to_num(m2_arr, nan=0.0, posinf=1.0, neginf=0.0)
        return EpisodeMetrics(
            loss_mean=float(loss_arr.mean()),
            loss_std=float(loss_arr.std(ddof=0)) if loss_arr.size > 1 else 0.0,
            metric1_mean=float(m1_arr.mean()),
            metric2_mean=float(m2_arr.mean()),
            metric1_std=float(m1_arr.std(ddof=0)) if m1_arr.size > 1 else 0.0,
            metric2_std=float(m2_arr.std(ddof=0)) if m2_arr.size > 1 else 0.0,
            loss_max=float(loss_arr.max()),
            metric1_min=float(m1_arr.min()),
            metric2_min=float(m2_arr.min()),
        )

    def _placeholder_metrics_for_invalid(self) -> EpisodeMetrics:
        placeholder_metric1 = float(self.baseline.metric1_mean or 0.0)
        if placeholder_metric1 < float(self.acc_threshold):
            placeholder_metric1 = float(self.acc_threshold)
        placeholder_metric2 = float(self.baseline.metric2_mean or 0.0)
        if self.acc_threshold_m2 is not None and placeholder_metric2 < float(self.acc_threshold_m2):
            placeholder_metric2 = float(self.acc_threshold_m2)
        placeholder_loss_std = float(self.baseline.loss_std or 0.0)
        if placeholder_loss_std > float(self.stab_threshold):
            placeholder_loss_std = float(self.stab_threshold)
        placeholder_loss_mean = float(self.baseline.loss_mean or 0.0)
        placeholder_m1_std = float(self.baseline.metric1_std or 0.0)
        placeholder_m2_std = float(self.baseline.metric2_std or 0.0)
        return EpisodeMetrics(
            loss_mean=placeholder_loss_mean,
            loss_std=placeholder_loss_std,
            metric1_mean=placeholder_metric1,
            metric2_mean=placeholder_metric2,
            metric1_std=placeholder_m1_std,
            metric2_std=placeholder_m2_std,
            loss_max=placeholder_loss_mean,
            metric1_min=placeholder_metric1,
            metric2_min=placeholder_metric2,
        )

    def prepare_action_for_terminal_probe(self, action_vec: np.ndarray) -> Dict[str, Any]:
        """Prepare optimizer-adjusted cfgs for a terminal reward probe.

        This mirrors the pre-forward part of :meth:`step` and exists so the
        sequential trainer can batch several completed actions onto different
        GPUs before running model-forward reward probes.
        """
        action_vec = np.asarray(action_vec, dtype=int).reshape(-1)
        if action_vec.size != self.total_action_dim:
            raise ValueError(
                f"action_vec dim {action_vec.size} != expected {self.total_action_dim}"
            )
        is_optimizer_baseline_action = bool(
            np.array_equal(action_vec, make_all_max_action_vector(self.num_layers))
        )
        action_vec_hash = action_hash(action_vec)
        degree_sync = self.sync_degree_vectors_from_model()
        timing: Dict[str, float] = {}
        cost_t0 = time.perf_counter()
        cost_eval = evaluate_action_for_cost(
            action_vec,
            profile=self.env_cfg.profile,
            num_layers=self.num_layers,
            max_sfs=self.max_sfs,
            rescale_bridge=self.rescale_bridge,
            gelu_degree=self.gelu_degree,
            attn_degree=self.attn_degree,
        )
        timing["cost_eval_wall_seconds"] = float(time.perf_counter() - cost_t0)
        decoded = cost_eval.decoded
        cfgs_dict = cost_eval.cfgs_dict
        opt_outputs = cost_eval.outputs
        opt_signals = cost_eval.signals
        any_invalid = bool(opt_signals.any_invalid)
        optimizer_invalid_summary = (
            summarize_optimizer_invalid_outputs(opt_outputs) if any_invalid else ""
        )

        bridge_invoker = getattr(self.rescale_bridge, "invoker", None)
        invoker_baselines: Mapping[str, Any] = getattr(bridge_invoker, "baselines", {}) or {}
        per_config_overrides: Dict[str, List[Tuple[str, str, Any, Any]]] = {}
        if not any_invalid:
            for cn, out in opt_outputs.items():
                try:
                    block_idx, _profile, layer_idx = parse_config_name(cn)
                except Exception:
                    continue
                if layer_idx < 0:
                    continue
                target_cfg = cfgs_dict[f"block{block_idx}"][int(layer_idx)]
                graph_key, _ = _strip_layer_suffix(cn)
                baseline_entry = invoker_baselines.get(graph_key)
                baseline_skeleton = list(baseline_entry[0]) if baseline_entry else []
                rotation_name_map = (self.env_cfg.rotation_name_map or {}).get(
                    (int(block_idx), str(self.env_cfg.profile)), {}
                )
                overrides = apply_optimizer_output_to_cfg(
                    target_cfg,
                    output_raw=out.raw,
                    block_idx=int(block_idx),
                    graph_key=graph_key,
                    baseline_skeleton=baseline_skeleton,
                    rotation_name_map=rotation_name_map,
                )
                if int(block_idx) == 2:
                    overrides = list(overrides) + sync_block2_qk_binding(target_cfg)
                    overrides = list(overrides) + sync_block2_aux_fresh_binding(target_cfg)
                elif int(block_idx) == 4:
                    overrides = list(overrides) + sync_block4_v_mask_binding(target_cfg)
                elif int(block_idx) == 5:
                    overrides = list(overrides) + sync_block5_aux_fresh_binding(target_cfg)
                if overrides:
                    per_config_overrides[cn] = [
                        (e.cfg_attr, e.source, e.old_value, e.new_value)
                        for e in overrides
                    ]

        info: Dict[str, Any] = {
            "decoded": decoded,
            "opt_signals": opt_signals,
            "opt_outputs_keys": list(opt_outputs.keys()),
            "invalid": any_invalid,
            "apply_failed": False,
            "eval_failed": False,
            "forward_ran": False,
            "optimizer_baseline_action": bool(is_optimizer_baseline_action),
            "optimizer_eval_mode": cost_eval.optimizer_eval_mode,
            "timing": timing,
        }
        if optimizer_invalid_summary:
            info["optimizer_invalid_summary"] = optimizer_invalid_summary
        if degree_sync:
            info["model_degree_sync"] = degree_sync
        if per_config_overrides:
            info["optimizer_cfg_overrides"] = per_config_overrides

        return {
            "action_vec": action_vec,
            "action_hash": action_vec_hash,
            "decoded": decoded,
            "opt_signals": opt_signals,
            "any_invalid": any_invalid,
            "requires_forward": not any_invalid,
            "info": info,
            "timing": timing,
        }

    def _finish_prepared_terminal_probe(
            self,
            prepared: Mapping[str, Any],
            metrics: EpisodeMetrics,
            *,
            probe_diagnostics: Optional[Mapping[str, Any]] = None,
            forward_ran: bool = True,
            eval_error: Optional[str] = None,
            ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        action_vec = np.asarray(prepared["action_vec"], dtype=int).reshape(-1)
        opt_signals = prepared["opt_signals"]
        any_invalid = bool(prepared.get("any_invalid", False))
        action_vec_hash = str(prepared.get("action_hash", action_hash(action_vec)))
        info = dict(prepared.get("info") or {})
        timing = dict(info.get("timing") or {})
        info["timing"] = timing
        info["forward_ran"] = bool(forward_ran)
        if eval_error:
            info["eval_failed"] = True
            info["error"] = eval_error
            any_invalid = True
        if any_invalid and not bool(forward_ran):
            info["forward_skipped_reason"] = info.get(
                "forward_skipped_reason", "any_invalid_chain"
            )
        if probe_diagnostics:
            diag = dict(probe_diagnostics)
            diag.update(timing)
            info["probe_diagnostics"] = diag

        breakdown = compute_reward(
            metrics, opt_signals,
            action_avg_k=avg_truncation_k_in_action(action_vec, self.num_layers),
            baseline=self.baseline,
            weights=self.reward_weights,
            acc_threshold=self.acc_threshold,
            acc_threshold_m2=self.acc_threshold_m2,
            stab_threshold=self.stab_threshold,
            any_invalid=any_invalid,
            pareto_archive=self.pareto_cost_archive,
            action_hash=action_vec_hash,
        )
        info["reward_breakdown"] = breakdown
        info["action_hash"] = action_vec_hash
        info["metrics"] = metrics
        info["invalid"] = bool(any_invalid)

        self._step_idx += 1
        self._last_invalid_rate = 1.0 if any_invalid else 0.0
        self._last_total_bits_norm = float(opt_signals.total_bits_sum) / max(
            1.0, float(self.baseline.total_bits_sum)
        )
        self._last_fusion_count = float(opt_signals.total_fusion_count)
        return self._build_state(), float(breakdown.reward), True, info

    def evaluate_prepared_terminal_batch(
            self,
            prepared_list: Sequence[Mapping[str, Any]],
            *,
            num_trials_per_action: int = 1,
            validation_required: bool = False,
            ) -> List[Tuple[np.ndarray, float, bool, Dict[str, Any]]]:
        """Evaluate prepared terminal candidates, batching distinct actions when possible."""
        prepared_items = list(prepared_list)
        if not prepared_items:
            return []
        k = max(1, int(num_trials_per_action))
        out: List[Optional[Tuple[np.ndarray, float, bool, Dict[str, Any]]]] = [None] * len(prepared_items)
        forward_indices = [
            i for i, item in enumerate(prepared_items)
            if bool(item.get("requires_forward", True))
        ]
        for i, item in enumerate(prepared_items):
            if i in forward_indices:
                continue
            metrics = self._placeholder_metrics_for_invalid()
            out[i] = self._finish_prepared_terminal_probe(
                item, metrics,
                probe_diagnostics={"fast_reward_mode": True, "validation_required": bool(validation_required)},
                forward_ran=False,
            )

        if forward_indices and self.probe_runner is not None and k == 1 and len(forward_indices) >= 2:
            base_seed = self._derive_probe_base_seed()
            self._probe_eval_counter += 1
            decoded_by_trial = [
                prepared_items[i]["decoded"] for i in forward_indices
            ]
            try:
                trial_results = self.probe_runner.run_action_trials_once(
                    decoded_by_trial, base_seed=base_seed,
                )
                diag_obj = self.probe_runner.last_diagnostics
                diag = {}
                if diag_obj is not None:
                    diag = {
                        "k": int(diag_obj.k),
                        "wall_seconds": float(diag_obj.wall_seconds),
                        "per_worker_seconds": [float(x) for x in diag_obj.per_worker_seconds],
                        "per_worker_trial_counts": [int(x) for x in diag_obj.per_worker_trial_counts],
                        "per_worker_trial_indices": [
                            list(map(int, x)) for x in diag_obj.per_worker_trial_indices
                        ],
                        "per_worker_trial_seeds": [
                            list(map(int, x)) for x in diag_obj.per_worker_trial_seeds
                        ],
                        "devices": [str(x) for x in diag_obj.devices],
                        "speedup_vs_sequential": float(diag_obj.speedup_vs_sequential),
                        "multi_action": bool(getattr(diag_obj, "multi_action", False)),
                        "line": format_diagnostics_line(diag_obj),
                    }
                diag.update({
                    "fast_reward_mode": True,
                    "online_num_trials_per_step": int(k),
                    "terminal_eval_batch_size": int(len(forward_indices)),
                    "validation_required": bool(validation_required),
                    "probe_install_wall_seconds": float(diag.get("wall_seconds", 0.0) or 0.0),
                    "probe_install_skipped": False,
                    "probe_clear_wall_seconds": 0.0,
                    "probe_clear_skipped": True,
                })
                for local_idx, result in enumerate(trial_results):
                    item_idx = forward_indices[local_idx]
                    metrics = self._metrics_from_trial_results([result])
                    out[item_idx] = self._finish_prepared_terminal_probe(
                        prepared_items[item_idx], metrics,
                        probe_diagnostics=diag,
                        forward_ran=True,
                    )
            except Exception as exc:
                for item_idx in forward_indices:
                    metrics = EpisodeMetrics(loss_mean=float("inf"), loss_std=float("inf"))
                    out[item_idx] = self._finish_prepared_terminal_probe(
                        prepared_items[item_idx], metrics,
                        probe_diagnostics={"fast_reward_mode": True, "validation_required": bool(validation_required)},
                        forward_ran=False,
                        eval_error=f"BLB multi-action eval failed: {exc}",
                    )
        else:
            for item_idx in forward_indices:
                item = prepared_items[item_idx]
                timing = dict((item.get("info") or {}).get("timing") or {})
                decoded = item["decoded"]
                action_vec_hash = str(item["action_hash"])
                persistent_install = bool(getattr(self.env_cfg, "persistent_probe_install", False))
                install_skipped = False
                install_t0 = time.perf_counter()
                try:
                    if persistent_install and self._installed_action_hash == action_vec_hash:
                        install_skipped = True
                    elif self.probe_runner is not None:
                        self.probe_runner.install_action(decoded)
                        self._installed_action_hash = action_vec_hash if persistent_install else None
                    else:
                        self.bridge.apply(
                            block1_cfgs=decoded.block1_cfgs,
                            block2_cfgs=decoded.block2_cfgs,
                            block3_cfgs=decoded.block3_cfgs,
                            block4_cfgs=decoded.block4_cfgs,
                            block5_cfgs=decoded.block5_cfgs,
                        )
                        self._installed_action_hash = action_vec_hash if persistent_install else None
                    timing["probe_install_wall_seconds"] = float(time.perf_counter() - install_t0)
                    timing["probe_install_skipped"] = float(1.0 if install_skipped else 0.0)
                    metrics = self._eval_on_probe(k)
                    diag = dict(self._last_probe_diagnostics or {})
                    diag.update(timing)
                    diag["fast_reward_mode"] = True
                    diag["online_num_trials_per_step"] = int(k)
                    diag["terminal_eval_batch_size"] = 1
                    diag["validation_required"] = bool(validation_required)
                    clear_t0 = time.perf_counter()
                    if persistent_install:
                        timing["probe_clear_wall_seconds"] = 0.0
                        timing["probe_clear_skipped"] = 1.0
                    else:
                        self.clear_installed_blb()
                        timing["probe_clear_wall_seconds"] = float(time.perf_counter() - clear_t0)
                        timing["probe_clear_skipped"] = 0.0
                    diag.update(timing)
                    out[item_idx] = self._finish_prepared_terminal_probe(
                        item, metrics,
                        probe_diagnostics=diag,
                        forward_ran=True,
                    )
                except Exception as exc:
                    try:
                        self.clear_installed_blb()
                    except Exception:
                        pass
                    metrics = EpisodeMetrics(loss_mean=float("inf"), loss_std=float("inf"))
                    out[item_idx] = self._finish_prepared_terminal_probe(
                        item, metrics,
                        probe_diagnostics={
                            "fast_reward_mode": True,
                            "validation_required": bool(validation_required),
                        },
                        forward_ran=False,
                        eval_error=f"BLB eval failed: {exc}",
                    )

        return [x for x in out if x is not None]

    def step(
            self,
            action_vec: np.ndarray,
            *,
            external_cost_score: Optional[float] = None,
            external_cost_rank: Optional[float] = None,
            ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """单步 episode：装噪声 → forward → 计算 reward → 还原噪声。

        ``external_cost_score`` / ``external_cost_rank``：fusion-count 路径专用。
        sequential_env 终局把 per-block 加权 cost 节省算好传进来，只在最终 valid
        reward（P3）里替掉聚合 fusion/K/bits cost。非 fusion 调用保持 None ⇒ 旧路径。
        invalid 分支必为 P1，compute_reward 不读 cost，故无需透传。
        """
        action_vec = np.asarray(action_vec, dtype=int).reshape(-1)
        if action_vec.size != self.total_action_dim:
            raise ValueError(
                f"action_vec dim {action_vec.size} != expected {self.total_action_dim}"
            )
        is_optimizer_baseline_action = bool(
            np.array_equal(action_vec, make_all_max_action_vector(self.num_layers))
        )
        action_vec_hash = action_hash(action_vec)

        degree_sync = self.sync_degree_vectors_from_model()
        timing: Dict[str, float] = {}
        cost_t0 = time.perf_counter()
        cost_eval = evaluate_action_for_cost(
            action_vec,
            profile=self.env_cfg.profile,
            num_layers=self.num_layers,
            max_sfs=self.max_sfs,
            rescale_bridge=self.rescale_bridge,
            gelu_degree=self.gelu_degree,
            attn_degree=self.attn_degree,
        )
        timing["cost_eval_wall_seconds"] = float(time.perf_counter() - cost_t0)
        decoded = cost_eval.decoded

        # 1) 调 Rescale_optimizer 拿 cost 信号
        cfgs_dict = cost_eval.cfgs_dict

        opt_outputs = cost_eval.outputs
        opt_signals = cost_eval.signals
        any_invalid = bool(opt_signals.any_invalid)
        optimizer_invalid_summary = (
            summarize_optimizer_invalid_outputs(opt_outputs) if any_invalid else ""
        )

        # 2) Optimizer-driven cfg override: rewrite every (block, layer) cfg in
        #    place so that the noise actually installed below reflects
        #    Rescale_optimizer's chosen SFs / fused rescales / effective
        #    rotations, rather than only the RL action's raw proposal.
        bridge_invoker = getattr(self.rescale_bridge, "invoker", None)
        invoker_baselines: Mapping[str, Any] = getattr(bridge_invoker, "baselines", {}) or {}
        per_config_overrides: Dict[str, List[Tuple[str, str, Any, Any]]] = {}
        if not any_invalid:
            for cn, out in opt_outputs.items():
                try:
                    block_idx, _profile, layer_idx = parse_config_name(cn)
                except Exception:
                    continue
                if layer_idx < 0:
                    continue
                target_cfg = cfgs_dict[f"block{block_idx}"][int(layer_idx)]
                graph_key, _ = _strip_layer_suffix(cn)
                baseline_entry = invoker_baselines.get(graph_key)
                # invoker.baselines is the InProcessInvoker tuple form
                # (skeleton, t_baseline, q_bits_baseline)
                baseline_skeleton = list(baseline_entry[0]) if baseline_entry else []
                rotation_name_map = (self.env_cfg.rotation_name_map or {}).get(
                    (int(block_idx), str(self.env_cfg.profile)), {}
                )
                overrides = apply_optimizer_output_to_cfg(
                    target_cfg,
                    output_raw=out.raw,
                    block_idx=int(block_idx),
                    graph_key=graph_key,
                    baseline_skeleton=baseline_skeleton,
                    rotation_name_map=rotation_name_map,
                )
                # Block 2 has Q/K binding (action_space writes wk_sf to both
                # wk_encode and wq_encode). The optimizer-driven override above
                # only refreshes the K-side encodes (those are the names in
                # GRAPH_NODE_TO_CFG_ATTR[2]); without this sync the model would
                # install Q-channel noise at the pre-override RL SF while
                # K-channel uses the optimizer-snapped SF.
                if int(block_idx) == 2:
                    # Q/K mask binding (action_space writes wk_sf to both
                    # K-side and Q-side encodes; optimizer write-back only
                    # refreshes K-side).
                    overrides = list(overrides) + sync_block2_qk_binding(target_cfg)
                    # x_centered_fresh / inv_std_fresh "x2" binding (action_space
                    # writes inv_std_fresh.sf to both; optimizer write-back only
                    # refreshes inv_std_fresh — apply_optimizer_output_to_cfg's
                    # SOURCE entry for block2 is cfg.inv_std_fresh).
                    overrides = list(overrides) + sync_block2_aux_fresh_binding(target_cfg)
                # Block 4 has mask2 binding (action_space writes the shared
                # softmax_out_mask SF to both softmax_out_mask_encode and
                # v_mask_encode). The optimizer-driven override only refreshes
                # softmax_out_mask_encode (the entry in GRAPH_NODE_TO_CFG_ATTR[4]
                # for ``ctpt_mask2``); without this sync, V-side install + the
                # ``ctct_rot_softmax_mul_v`` delta computation (which reads
                # cfg.v_mask_encode.scaling_factor) would drift from the
                # softmax_out side.
                elif int(block_idx) == 4:
                    overrides = list(overrides) + sync_block4_v_mask_binding(target_cfg)
                # Block 5 mirror of Block 2: x_centered_fresh / inv_std_fresh
                # "x2" binding. The SOURCE entry for block5 is cfg.x_centered_fresh,
                # so the optimizer refreshes that and we mirror onto inv_std_fresh.
                elif int(block_idx) == 5:
                    overrides = list(overrides) + sync_block5_aux_fresh_binding(target_cfg)
                if overrides:
                    per_config_overrides[cn] = [
                        (e.cfg_attr, e.source, e.old_value, e.new_value)
                        for e in overrides
                    ]

        # 3) 基础诊断信息。Rescale_optimizer 的 invalid 是成本/可行性信号，
        #    但不能再作为跳过模型 forward 的理由；终端 reward 必须看到
        #    实际安装后的 probe metrics / stability。
        info: Dict[str, Any] = {
            "decoded": decoded,
            "opt_signals": opt_signals,
            "opt_outputs_keys": list(opt_outputs.keys()),
            "invalid": any_invalid,
            "apply_failed": False,
            "eval_failed": False,
            "forward_ran": False,
            "optimizer_baseline_action": bool(is_optimizer_baseline_action),
            "optimizer_eval_mode": cost_eval.optimizer_eval_mode,
            "timing": timing,
        }
        if optimizer_invalid_summary:
            info["optimizer_invalid_summary"] = optimizer_invalid_summary
        if degree_sync:
            info["model_degree_sync"] = degree_sync
        if per_config_overrides:
            info["optimizer_cfg_overrides"] = per_config_overrides

        # 3.5) Short-circuit: if Rescale_optimizer already marked any block as
        # invalid_chain, the assembled cfg is incoherent — installing it would
        # produce NaN/inf logits and the model forward is wasted compute.
        # Skip steps 4–6 entirely; emit a priority-3 cost-only reward (with the
        # invalid_penalty docked) using baseline-derived placeholder metrics.
        # This is what the user asked for on 2026-05-17: "出现invalid chain
        # 再去做推理就没有意义了，不用再去做推理了".
        #
        # Placeholder metrics MUST clear the acc / stab gates so the reward
        # ``priority`` label reflects the actual failure mode (invalid_chain →
        # cost-layer priority=3) rather than a spurious acc-violation triggered
        # by the placeholder defaults. We use the noisy baseline metric (if the
        # runner has calibrated one) and otherwise fall back to the threshold
        # value itself, which leaves ``acc_violation = stab_excess = 0`` so the
        # only reward contribution from the gate path is ``invalid_term``.
        if any_invalid:
            placeholder_metric1 = float(self.baseline.metric1_mean or 0.0)
            if placeholder_metric1 < float(self.acc_threshold):
                placeholder_metric1 = float(self.acc_threshold)
            placeholder_metric2 = float(self.baseline.metric2_mean or 0.0)
            if self.acc_threshold_m2 is not None and placeholder_metric2 < float(self.acc_threshold_m2):
                placeholder_metric2 = float(self.acc_threshold_m2)
            placeholder_loss_std = float(self.baseline.loss_std or 0.0)
            if placeholder_loss_std > float(self.stab_threshold):
                placeholder_loss_std = float(self.stab_threshold)
            placeholder_loss_mean = float(self.baseline.loss_mean or 0.0)
            # v3 stability path: m1_std/m2_std also enter combined_stab_excess.
            # Use baseline stds (which v3 thresholds will treat as "passing" by
            # construction) so the short-circuit can't trip stab_violation.
            placeholder_m1_std = float(self.baseline.metric1_std or 0.0)
            placeholder_m2_std = float(self.baseline.metric2_std or 0.0)
            metrics = EpisodeMetrics(
                loss_mean=placeholder_loss_mean,
                loss_std=placeholder_loss_std,
                metric1_mean=placeholder_metric1,
                metric2_mean=placeholder_metric2,
                metric1_std=placeholder_m1_std,
                metric2_std=placeholder_m2_std,
                loss_max=placeholder_loss_mean,
                metric1_min=placeholder_metric1,
                metric2_min=placeholder_metric2,
            )
            breakdown = compute_reward(
                metrics, opt_signals,
                action_avg_k=avg_truncation_k_in_action(action_vec, self.num_layers),
                baseline=self.baseline,
                weights=self.reward_weights,
                acc_threshold=self.acc_threshold,
                acc_threshold_m2=self.acc_threshold_m2,
                stab_threshold=self.stab_threshold,
                any_invalid=True,
                pareto_archive=self.pareto_cost_archive,
                action_hash=action_vec_hash,
            )
            info["reward_breakdown"] = breakdown
            info["action_hash"] = action_vec_hash
            info["metrics"] = metrics
            info["forward_ran"] = False
            info["forward_skipped_reason"] = "any_invalid_chain"
            info["invalid"] = True
            self._step_idx += 1
            self._last_invalid_rate = 1.0
            self._last_total_bits_norm = float(opt_signals.total_bits_sum) / max(
                1.0, float(self.baseline.total_bits_sum)
            )
            self._last_fusion_count = float(opt_signals.total_fusion_count)
            return self._build_state(), float(breakdown.reward), True, info

        # 4) 装 BLB 噪声
        # 语义更新（2026-05）：first_input fresh 噪声不再注入（"第一个 HE 配置
        # 无损"），且 layer-0 block1 整体不安装。decoded.block1_cfgs 已不含
        # layer 0；first_input_sf 字段保留为占位 0 不传给 bridge。
        # When probe_runner is set (multi-GPU), install on every worker so each
        # GPU's model carries the same BLB cfg before its trial subset runs.
        persistent_install = bool(getattr(self.env_cfg, "persistent_probe_install", False))
        install_skipped = False
        install_t0 = time.perf_counter()
        try:
            if persistent_install and self._installed_action_hash == action_vec_hash:
                install_skipped = True
            elif self.probe_runner is not None:
                self.probe_runner.install_action(decoded)
                self._installed_action_hash = action_vec_hash if persistent_install else None
            else:
                self.bridge.apply(
                    block1_cfgs=decoded.block1_cfgs,
                    block2_cfgs=decoded.block2_cfgs,
                    block3_cfgs=decoded.block3_cfgs,
                    block4_cfgs=decoded.block4_cfgs,
                    block5_cfgs=decoded.block5_cfgs,
                )
                self._installed_action_hash = action_vec_hash if persistent_install else None
        except Exception as exc:
            try:
                self.clear_installed_blb()
            except Exception:
                pass
            timing["probe_install_wall_seconds"] = float(time.perf_counter() - install_t0)
            timing["probe_install_skipped"] = float(0.0)
            # 互斥校验失败，按 invalid 处理
            metrics = EpisodeMetrics(loss_mean=float("inf"), loss_std=float("inf"))
            breakdown = compute_reward(
                metrics, opt_signals,
                action_avg_k=avg_truncation_k_in_action(action_vec, self.num_layers),
                baseline=self.baseline,
                weights=self.reward_weights,
                acc_threshold=self.acc_threshold,
                acc_threshold_m2=self.acc_threshold_m2,
                stab_threshold=self.stab_threshold,
                any_invalid=True,
                pareto_archive=self.pareto_cost_archive,
                action_hash=action_vec_hash,
            )
            info["reward_breakdown"] = breakdown
            info["action_hash"] = action_vec_hash
            info["error"] = f"BLB apply failed: {exc}"
            info["invalid"] = True
            info["apply_failed"] = True
            info["timing"] = timing
            info["metrics"] = metrics
            self._step_idx += 1
            self._last_invalid_rate = 1.0
            self._last_total_bits_norm = float(opt_signals.total_bits_sum) / max(1.0, float(self.baseline.total_bits_sum))
            self._last_fusion_count = float(opt_signals.total_fusion_count)
            return self._build_state(), float(breakdown.reward), True, info
        timing["probe_install_wall_seconds"] = float(time.perf_counter() - install_t0)
        timing["probe_install_skipped"] = float(1.0 if install_skipped else 0.0)

        # 5) forward + metrics（多 trial）
        try:
            metrics = self._eval_on_probe(self.env_cfg.num_trials_per_step)
            info["forward_ran"] = True
            metrics = self._maybe_borderline_retest(metrics, info)
            if self._last_probe_diagnostics:
                diag = dict(self._last_probe_diagnostics)
                diag.update(timing)
                diag["persistent_probe_install"] = bool(persistent_install)
                diag["probe_install_skipped"] = bool(install_skipped)
                info["probe_diagnostics"] = diag
        except Exception as exc:
            try:
                self.clear_installed_blb()
            except Exception:
                pass
            metrics = EpisodeMetrics(loss_mean=float("inf"), loss_std=float("inf"))
            breakdown = compute_reward(
                metrics, opt_signals,
                action_avg_k=avg_truncation_k_in_action(action_vec, self.num_layers),
                baseline=self.baseline,
                weights=self.reward_weights,
                acc_threshold=self.acc_threshold,
                acc_threshold_m2=self.acc_threshold_m2,
                stab_threshold=self.stab_threshold,
                any_invalid=True,
                pareto_archive=self.pareto_cost_archive,
                action_hash=action_vec_hash,
            )
            info["reward_breakdown"] = breakdown
            info["action_hash"] = action_vec_hash
            info["error"] = f"BLB eval failed: {exc}"
            info["invalid"] = True
            info["eval_failed"] = True
            info["timing"] = timing
            info["metrics"] = metrics
            self._step_idx += 1
            self._last_invalid_rate = 1.0
            self._last_total_bits_norm = float(opt_signals.total_bits_sum) / max(1.0, float(self.baseline.total_bits_sum))
            self._last_fusion_count = float(opt_signals.total_fusion_count)
            return self._build_state(), float(breakdown.reward), True, info
        else:
            clear_t0 = time.perf_counter()
            if persistent_install:
                timing["probe_clear_wall_seconds"] = 0.0
                timing["probe_clear_skipped"] = 1.0
            else:
                self.clear_installed_blb()
                timing["probe_clear_wall_seconds"] = float(time.perf_counter() - clear_t0)
                timing["probe_clear_skipped"] = 0.0
            if "probe_diagnostics" in info:
                diag = dict(info["probe_diagnostics"])
                diag.update(timing)
                diag["probe_clear_skipped"] = bool(persistent_install)
                info["probe_diagnostics"] = diag

        # 6) reward
        breakdown = compute_reward(
            metrics, opt_signals,
            action_avg_k=avg_truncation_k_in_action(action_vec, self.num_layers),
            baseline=self.baseline,
            weights=self.reward_weights,
            acc_threshold=self.acc_threshold,
            acc_threshold_m2=self.acc_threshold_m2,
            stab_threshold=self.stab_threshold,
            any_invalid=any_invalid,
            pareto_archive=self.pareto_cost_archive,
            action_hash=action_vec_hash,
            external_cost_score=external_cost_score,
            external_cost_rank=external_cost_rank,
        )

        info["reward_breakdown"] = breakdown
        info["action_hash"] = action_vec_hash
        info["metrics"] = metrics

        # state 更新
        self._step_idx += 1
        self._last_invalid_rate = 1.0 if any_invalid else 0.0
        self._last_total_bits_norm = float(opt_signals.total_bits_sum) / max(1.0, float(self.baseline.total_bits_sum))
        self._last_fusion_count = float(opt_signals.total_fusion_count)

        return self._build_state(), float(breakdown.reward), True, info

    # ------------------------------------------------------------------
    # Multi-GPU probe seed derivation
    # ------------------------------------------------------------------
    def _derive_probe_base_seed(self) -> int:
        """Per-action probe seed for the ProbeRunner.

        Built from (step_idx, action-eval counter, wall ns). Reproducible
        across reruns if step_idx/counter are reset; deterministic across
        the two GPU workers (they get the same base_seed and the per-trial
        offset comes from ``trial_idx`` inside ``_trial_seed``).
        """
        return int(
            (
                (int(self._step_idx) * 1_000_003)
                ^ (int(self._probe_eval_counter) * 2654435761)
                ^ (int(time.time_ns()) & 0xFFFFFFFF)
            ) & 0x7FFFFFFFFFFFFFFF
        )

    # ------------------------------------------------------------------
    # 评估子集 forward（单层 K-trials）
    # ------------------------------------------------------------------
    def _acc_worst_deficit_norm(self, metrics: EpisodeMetrics) -> float:
        """Worst per-channel accuracy deficit normalized by |baseline - thr|.

        0.0 when both gates pass (or thresholds aren't calibrated yet, e.g.
        during the noisy baseline preflight). Mirrors the ADR-012 near-miss
        coordinate in reward.compute_reward.
        """
        worst = 0.0
        floor = float(getattr(self.reward_weights, "margin_denom_floor", 1e-6) or 1e-6)
        for m, thr, base in (
                (float(metrics.metric1_mean),
                 self.acc_threshold,
                 float(getattr(self.reward_weights, "baseline_metric1", 0.0) or 0.0)),
                (float(metrics.metric2_mean),
                 self.acc_threshold_m2,
                 float(getattr(self.reward_weights, "baseline_metric2", 0.0) or 0.0)),
        ):
            if thr is None:
                continue
            thr_f = float(thr)
            if not (math.isfinite(thr_f) and math.isfinite(m) and abs(base) > floor):
                continue
            denom = max(abs(base - thr_f), floor)
            d = (thr_f - m) / denom
            if d > 0.0:
                worst = max(worst, d)
        return float(worst)

    def _maybe_borderline_retest(
            self,
            metrics: EpisodeMetrics,
            info: Dict[str, Any],
            ) -> EpisodeMetrics:
        """ADR-012: re-measure borderline metric fails with fresh trials.

        With a 256-sample probe, m1 is quantized (~0.004/sample, std ~0.0018):
        a config whose TRUE accuracy is within tolerance still lands below the
        threshold a few % of the time, eating the P1 hammer. One fresh
        re-measurement at 2x trials drops that false-fail rate quadratically
        while true violators keep failing. The retest verdict REPLACES the
        first measurement. Deterministic: the retest probe seed is a salt of
        the episode-keyed probe_noise_seed, so it is identical for any GPU
        count (1==N preserved). No-op when the deficit is zero, beyond the
        near-miss band, on the legacy random-probe path, or when disabled.
        """
        if not bool(getattr(self.env_cfg, "borderline_retest_enabled", False)):
            return metrics
        if self.probe_noise_seed is None or self.probe_runner is not None:
            return metrics
        band = float(getattr(self.reward_weights, "near_miss_band", 0.0) or 0.0)
        if band <= 0.0:
            return metrics
        deficit = self._acc_worst_deficit_norm(metrics)
        if deficit <= 0.0 or deficit > band:
            return metrics
        mult = max(1, int(getattr(self.env_cfg, "borderline_retest_trials_multiplier", 2)))
        retest_k = mult * max(1, int(self.env_cfg.num_trials_per_step))
        first_seed = int(self.probe_noise_seed)
        # Golden-ratio salt -> a disjoint deterministic trial-seed stream.
        self.probe_noise_seed = int(
            (first_seed ^ 0x9E3779B97F4A7C15) & 0x7FFFFFFFFFFFFFFF
        )
        try:
            retest_metrics = self._eval_on_probe(retest_k)
        finally:
            self.probe_noise_seed = first_seed
        info["borderline_retest"] = {
            "first_metric1_mean": float(metrics.metric1_mean),
            "first_metric2_mean": float(metrics.metric2_mean),
            "first_deficit_norm": float(deficit),
            "retest_trials": int(retest_k),
            "retest_metric1_mean": float(retest_metrics.metric1_mean),
            "retest_metric2_mean": float(retest_metrics.metric2_mean),
            "retest_deficit_norm": float(self._acc_worst_deficit_norm(retest_metrics)),
        }
        return retest_metrics

    def _eval_on_probe(self, k_trials: int) -> EpisodeMetrics:
        """在 ``self.probe_batches`` 上跑 k_trials 次（独立 RNG），返回 EpisodeMetrics。

        If ``self.probe_runner`` is set (multi-GPU), trials are split round-robin
        across workers and run in parallel threads (one per GPU). Otherwise the
        original sequential single-GPU path runs unchanged.
        """
        k = max(1, int(k_trials))

        # Deterministic episode-parallel path: keyed noise, own-device only,
        # NO global RNG save/restore (set_rng_state_all from concurrent worker
        # threads would clobber sibling workers' freshly-seeded generators).
        if self.probe_noise_seed is not None and self.probe_runner is None:
            return self._eval_on_probe_deterministic(k)

        # 保存外层 RNG 状态以避免污染
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        np_rng = np.random.get_state()

        per_trial_loss: List[float] = []
        per_trial_metric1: List[float] = []
        per_trial_metric2: List[float] = []
        probe_wall_start = time.perf_counter()

        try:
            if self.probe_runner is not None:
                # ---- Multi-GPU path: fan out via ProbeRunner ----
                base_seed = self._derive_probe_base_seed()
                self._probe_eval_counter += 1
                results = self.probe_runner.run_trials(k, base_seed=base_seed)
                diag = self.probe_runner.last_diagnostics
                if diag is not None:
                    self._last_probe_diagnostics = {
                        "k": int(diag.k),
                        "wall_seconds": float(diag.wall_seconds),
                        "per_worker_seconds": [float(x) for x in diag.per_worker_seconds],
                        "per_worker_trial_counts": [int(x) for x in diag.per_worker_trial_counts],
                        "per_worker_trial_indices": [list(map(int, x)) for x in diag.per_worker_trial_indices],
                        "per_worker_trial_seeds": [list(map(int, x)) for x in diag.per_worker_trial_seeds],
                        "devices": [str(x) for x in diag.devices],
                        "speedup_vs_sequential": float(diag.speedup_vs_sequential),
                        "line": format_diagnostics_line(diag),
                    }
                for (loss, m1, m2) in results:
                    if loss is None or (isinstance(loss, float) and not math.isfinite(loss)):
                        # NaN/inf from a probe trial is kept and handled below
                        # via _LOSS_CAP clamping (same semantics as single-GPU).
                        per_trial_loss.append(float(loss) if loss is not None else float("nan"))
                    else:
                        per_trial_loss.append(float(loss))
                    per_trial_metric1.append(float(m1))
                    per_trial_metric2.append(float(m2))
            else:
                # ---- Single-GPU path (unchanged from pre-multi-GPU era) ----
                was_training = self.model.training
                self.model.eval()
                try:
                    with torch.inference_mode():
                        for trial_idx in range(k):
                            # 独立 trial seed —— 让噪声采样独立，但模型权重 / data 不变
                            seed = (int(time.time_ns()) ^ (trial_idx * 1_000_003)) & 0x7FFFFFFFFFFFFFFF
                            torch.manual_seed(seed)
                            np.random.seed(seed % (2**32))
                            if torch.cuda.is_available():
                                torch.cuda.manual_seed_all(seed)

                            losses, m1s, m2s = [], [], []
                            for batch in self.probe_batches:
                                kwargs: Dict[str, torch.Tensor] = {
                                    "input_ids": batch.input_ids,
                                    "attention_mask": batch.attention_mask,
                                    "labels": batch.labels,
                                }
                                if batch.token_type_ids is not None:
                                    kwargs["token_type_ids"] = batch.token_type_ids
                                outputs = self.model(**kwargs)
                                logits = outputs.logits if hasattr(outputs, "logits") else outputs[1]
                                loss, m1, m2 = _compute_metrics_on_batch(
                                    logits, batch.labels, is_regression=self.is_regression,
                                )
                                losses.append(loss)
                                m1s.append(m1)
                                m2s.append(m2)

                            if losses:
                                per_trial_loss.append(float(np.mean(losses)))
                                per_trial_metric1.append(float(np.mean(m1s)))
                                per_trial_metric2.append(float(np.mean(m2s)))
                    wall_elapsed = time.perf_counter() - probe_wall_start
                    self._last_probe_diagnostics = {
                        "k": int(k),
                        "wall_seconds": float(wall_elapsed),
                        "per_worker_seconds": [float(wall_elapsed)],
                        "per_worker_trial_counts": [int(k)],
                        "per_worker_trial_indices": [list(range(int(k)))],
                        "per_worker_trial_seeds": [],
                        "devices": [str(self._device)],
                        "speedup_vs_sequential": 1.0,
                        "line": (
                            f"[probe-runner] k={int(k)} split=[{int(k)}] "
                            f"devices=[{self._device}] wall={wall_elapsed:.3f}s "
                            f"worker_seconds=[{wall_elapsed:.3f}] speedup=1.00x "
                            f"trials=[{self._device}:{list(range(int(k)))}]"
                        ),
                    }
                finally:
                    if was_training:
                        self.model.train()
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            np.random.set_state(np_rng)

        return self._aggregate_probe_trials(
            per_trial_loss, per_trial_metric1, per_trial_metric2,
        )

    def _eval_on_probe_deterministic(self, k: int) -> EpisodeMetrics:
        """K serial trials on this env's device with keyed noise seeds.

        Trial ``t`` reseeds ONLY this device's dedicated noise generator with
        ``probe_noise_seed XOR (t * KNUTH)`` (the same mix as
        ``probe_runner._trial_seed``), so the injected CKKS/MPC noise depends
        on (run_seed, global_episode, trial) alone: identical on 1 GPU and on
        any worker of an N-GPU run (CUDA Philox is device-independent), and
        reproducible across reruns. Touches no global RNG → safe for
        concurrent episode-parallel workers.

        With workers-per-device > 1 the dedicated noise generator is shared
        by same-device siblings, so each trial's (reseed -> forward) runs
        under ``probe_device_lock`` — trials interleave across workers at
        trial granularity with identical per-trial noise streams regardless
        of interleaving order.
        """
        from function_handler import reseed_noise_rng_for_device

        lock = self.probe_device_lock
        if lock is None:
            lock = _NULL_CTX
        base_seed = int(self.probe_noise_seed)
        per_trial_loss: List[float] = []
        per_trial_metric1: List[float] = []
        per_trial_metric2: List[float] = []
        probe_wall_start = time.perf_counter()
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                for trial_idx in range(int(k)):
                    seed = int(
                        (base_seed ^ (trial_idx * 2654435761)) & 0x7FFFFFFFFFFFFFFF
                    )
                    with lock:
                        reseed_noise_rng_for_device(self._device, seed)

                        losses, m1s, m2s = [], [], []
                        for batch in self.probe_batches:
                            kwargs: Dict[str, torch.Tensor] = {
                                "input_ids": batch.input_ids,
                                "attention_mask": batch.attention_mask,
                                "labels": batch.labels,
                            }
                            if batch.token_type_ids is not None:
                                kwargs["token_type_ids"] = batch.token_type_ids
                            outputs = self.model(**kwargs)
                            logits = outputs.logits if hasattr(outputs, "logits") else outputs[1]
                            loss, m1, m2 = _compute_metrics_on_batch(
                                logits, batch.labels, is_regression=self.is_regression,
                            )
                            losses.append(loss)
                            m1s.append(m1)
                            m2s.append(m2)

                    if losses:
                        per_trial_loss.append(float(np.mean(losses)))
                        per_trial_metric1.append(float(np.mean(m1s)))
                        per_trial_metric2.append(float(np.mean(m2s)))
        finally:
            if was_training:
                self.model.train()
        wall_elapsed = time.perf_counter() - probe_wall_start
        self._last_probe_diagnostics = {
            "k": int(k),
            "wall_seconds": float(wall_elapsed),
            "per_worker_seconds": [float(wall_elapsed)],
            "per_worker_trial_counts": [int(k)],
            "per_worker_trial_indices": [list(range(int(k)))],
            "per_worker_trial_seeds": [[
                int((base_seed ^ (t * 2654435761)) & 0x7FFFFFFFFFFFFFFF)
                for t in range(int(k))
            ]],
            "devices": [str(self._device)],
            "speedup_vs_sequential": 1.0,
            "deterministic_probe_seed": int(base_seed),
            "line": (
                f"[probe-deterministic] k={int(k)} device={self._device} "
                f"base_seed={base_seed} wall={wall_elapsed:.3f}s"
            ),
        }
        return self._aggregate_probe_trials(
            per_trial_loss, per_trial_metric1, per_trial_metric2,
        )

    def _aggregate_probe_trials(
            self,
            per_trial_loss: List[float],
            per_trial_metric1: List[float],
            per_trial_metric2: List[float],
            ) -> EpisodeMetrics:
        if not per_trial_loss:
            return EpisodeMetrics()

        loss_arr = np.array(per_trial_loss, dtype=float)
        m1_arr = np.array(per_trial_metric1, dtype=float)
        m2_arr = np.array(per_trial_metric2, dtype=float)

        # Clamp non-finite cross-entropy outputs (heavy BLB noise can push some
        # trials to inf/nan via logit overflow). Without the clamp a *single*
        # overflowing trial would make np.std → inf, every action would land in
        # the same priority-2 fallback bucket (terminal_reward ≡ -150), and PPO
        # would see no gradient between candidates. Clamping to 100 (≫ a
        # normal MRPC cross_entropy in [0, 5]) preserves rank order while
        # keeping the std finite and comparable across actions.
        _LOSS_CAP = 100.0
        loss_arr = np.nan_to_num(loss_arr, nan=_LOSS_CAP, posinf=_LOSS_CAP, neginf=_LOSS_CAP)
        loss_arr = np.clip(loss_arr, 0.0, _LOSS_CAP)
        m1_arr = np.nan_to_num(m1_arr, nan=0.0, posinf=1.0, neginf=0.0)
        m2_arr = np.nan_to_num(m2_arr, nan=0.0, posinf=1.0, neginf=0.0)

        return EpisodeMetrics(
            loss_mean=float(loss_arr.mean()),
            loss_std=float(loss_arr.std(ddof=0)) if loss_arr.size > 1 else 0.0,
            metric1_mean=float(m1_arr.mean()),
            metric2_mean=float(m2_arr.mean()),
            metric1_std=float(m1_arr.std(ddof=0)) if m1_arr.size > 1 else 0.0,
            metric2_std=float(m2_arr.std(ddof=0)) if m2_arr.size > 1 else 0.0,
            loss_max=float(loss_arr.max()),
            metric1_min=float(m1_arr.min()),
            metric2_min=float(m2_arr.min()),
        )

    # ------------------------------------------------------------------
    # state 构造
    # ------------------------------------------------------------------
    def _build_state(self) -> np.ndarray:
        """spec §5.1 最小 state（带 per-layer indicator 占位）。

        组成：
          [softmax_degree, gelu_degree, num_layers,
           profile_id_hash_0..1, last_total_bits_norm, last_fusion_count_norm,
           last_invalid_rate, step_idx_norm,
           per_layer_step_indicator_0..L-1]
        """
        static = [
            float(self.attn_degree_state),
            float(self.gelu_degree_state),
            float(self.num_layers),
        ]
        # profile ID hash → 2 维 [-1, 1]
        h = abs(hash(self.env_cfg.profile)) & 0xFFFFFFFF
        prof = [
            ((h % 1009) / 1009.0) * 2.0 - 1.0,
            (((h // 1009) % 1009) / 1009.0) * 2.0 - 1.0,
        ]
        last = [
            float(self._last_total_bits_norm),
            float(self._last_fusion_count) / max(1.0, float(self.num_layers * 5)),
            float(self._last_invalid_rate),
            float(self._step_idx) / max(1.0, 100.0),
        ]
        per_layer = [(li % 12) / 12.0 for li in range(self.num_layers)]
        state = np.asarray(static + prof[:1] + last + per_layer, dtype=np.float32)
        # 保证 state_dim 一致
        target_dim = self.state_dim
        if state.shape[0] < target_dim:
            pad = np.zeros(target_dim - state.shape[0], dtype=np.float32)
            state = np.concatenate([state, pad], axis=0)
        elif state.shape[0] > target_dim:
            state = state[:target_dim]
        return state


# ---------------------------------------------------------------------------
# Baseline 计算辅助（spec §6.3）
# ---------------------------------------------------------------------------
def estimate_baseline_cost_stats(
        env: BLBStage2Env,
        sample_count: int = 1,
        *,
        precomputed_baseline_signals: Optional[Mapping[str, Any]] = None,
        ) -> BaselineCostStats:
    """基于 static_skeletons baseline 校准 reward 权重。

    baseline cost 必须由 Rescale_optimizer 的 static_skeletons archive 提供；
    本函数只跑若干 random action 估计典型 ``bits_drop`` / ``fusion_count`` /
    ``k_drop``，用于反推 reward 权重。

    Args:
        env:                          ``BLBStage2Env`` 实例。
        sample_count:                 估计 typical drop 的随机采样次数。
        precomputed_baseline_signals: 必填。必须来自 ``load_static_skeletons_baseline``
                                       / ``static_skeletons_baseline_to_action``；
                                       本函数只补 random-sample 部分来估计
                                       typical_*_drop。
    """
    env.sync_degree_vectors_from_model()

    if precomputed_baseline_signals is None:
        raise RuntimeError(
            "BLB Stage-2 baseline must come from "
            "Rescale_optimizer/configs/<dataset>/static_skeletons_<dataset>.json; "
            "refusing to estimate baseline from the all-max action path."
        )

    baseline_total_bits = int(precomputed_baseline_signals.get("total_bits_sum", 0))
    baseline_fusion_count = int(precomputed_baseline_signals.get("total_fusion_count", 0))
    baseline_avg_k = float(precomputed_baseline_signals.get("avg_k", max(K_LEVELS)))

    # 估计典型 drop —— 仍然走 RO 评估，因为我们需要"random action 相对 baseline 的 cost
    # 差"来反推 reward 权重。precomputed 路径下 baseline_total_bits 是权威值。
    bits_drops: List[float] = []
    fusion_counts: List[float] = []
    k_drops: List[float] = []
    rng = np.random.default_rng(seed=42)
    for _ in range(max(0, int(sample_count))):
        random_action = np.array(
            [rng.integers(0, d) for d in env.action_dims], dtype=int,
        )
        rd_eval = evaluate_action_for_cost(
            random_action,
            profile=env.env_cfg.profile,
            num_layers=env.num_layers,
            max_sfs=env.max_sfs,
            rescale_bridge=env.rescale_bridge,
            gelu_degree=env.gelu_degree,
            attn_degree=env.attn_degree,
        )
        rd_signals = rd_eval.signals
        if rd_signals.any_invalid:
            continue
        bits_drops.append(float(baseline_total_bits) - float(rd_signals.total_bits_sum))
        fusion_counts.append(float(rd_signals.total_fusion_count))
        avg_k = avg_truncation_k_in_action(random_action, env.num_layers)
        k_drops.append(float(baseline_avg_k) - float(avg_k))

    return BaselineCostStats(
        total_bits_sum=int(baseline_total_bits),
        total_fusion_count=int(baseline_fusion_count),
        avg_k=float(baseline_avg_k),
        typical_bits_drop=float(np.mean(bits_drops)) if bits_drops else 1.0,
        typical_fusion_count=float(np.mean(fusion_counts)) if fusion_counts else 1.0,
        typical_k_drop=float(np.mean(k_drops)) if k_drops else 1.0,
    )
