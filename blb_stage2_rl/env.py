"""BLB Stage 2 RL ``Env`` 包装。

不强依赖 ``gymnasium`` —— 我们只用最小化的 ``reset/step`` 接口，避免新增依赖。
"""
from __future__ import annotations

import copy
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from blb_rl_bridge import BLBNoiseRLBridge
from rescale_optimizer_bridge import (
    RescaleOptimizerBridge,
    aggregate_optimizer_signals,
    apply_rotation_flags_to_cfg,
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
from .default_invoker import HeuristicStubInvoker
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
@dataclass
class BLBStage2EnvConfig:
    """``BLBStage2Env`` 的运行参数。"""
    profile: str = "default"
    num_trials_per_step: int = 3            # spec §5.3 推荐 3 次取 std
    probe_batch_count: int = 4              # 每次 trial 跑多少 mini-batch
    deterministic_eval: bool = False
    rotation_name_map: Optional[Mapping[Tuple[int, str], Mapping[str, str]]] = None


class BLBStage2Env:
    """把"action → install BLB → forward → metrics → reward"封装成单步 env。

    单步 episode（horizon=1），每次 ``step(action)`` 返回 (state, reward, done=True, info)。

    依赖：
      * ``handler``：``ReversibleLayerHandler`` 实例（已经替换好 GELU/Softmax 近似）
      * ``model``：HF 模型（用于 forward）
      * ``probe_batches``：评估用的 mini-batches list（每条是 ProbeBatch）
      * ``rescale_bridge``：``RescaleOptimizerBridge``（HeuristicStubInvoker 兜底）
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
            heuristic_invoker: Optional[HeuristicStubInvoker] = None,
            stub_register_cfgs: bool = True,
            ):
        self.handler = handler
        self.model = model
        self.probe_batches = list(probe_batches)
        self.rescale_bridge = rescale_bridge
        self.baseline = baseline
        self.reward_weights = reward_weights
        self.acc_threshold = float(acc_threshold)
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
        self._heuristic = heuristic_invoker
        self._stub_register_cfgs = bool(stub_register_cfgs)

        self.bridge = BLBNoiseRLBridge(handler, layers_attribute=layers_attribute)

        self.action_dims = action_dims_for_config(self.num_layers)
        self.total_action_dim = len(self.action_dims)

        # state 设计：spec §5.1 minimal state
        self._last_total_bits_norm: float = 0.0
        self._last_fusion_count: float = 0.0
        self._last_invalid_rate: float = 0.0
        self._step_idx: int = 0

        self._device = next(model.parameters()).device

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

        # 2) 清掉本 env 之前可能装过的 BLB 噪声（重复 clear 安全）
        try:
            self.bridge.clear()
        except Exception:
            pass

        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed) % (2**32))
            random.seed(int(seed))

        self.sync_degree_vectors_from_model()
        return self._build_state()

    def step(
            self,
            action_vec: np.ndarray,
            ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """单步 episode：装噪声 → forward → 计算 reward → 还原噪声。"""
        action_vec = np.asarray(action_vec, dtype=int).reshape(-1)
        if action_vec.size != self.total_action_dim:
            raise ValueError(
                f"action_vec dim {action_vec.size} != expected {self.total_action_dim}"
            )

        degree_sync = self.sync_degree_vectors_from_model()
        decoded = action_vector_to_cfgs(
            action_vec=action_vec,
            max_sfs=self.max_sfs,
            num_layers=self.num_layers,
            gelu_degree=self.gelu_degree,
            attn_degree=self.attn_degree,
        )

        # 1) 调 Rescale_optimizer 拿 cost 信号
        cfgs_dict = decoded.cfgs_dict()
        opt_requests = build_optimizer_requests(self.env_cfg.profile, cfgs_dict)

        # 注册 cfg 到 heuristic invoker（后者从 cfg 抽 SF）
        if self._heuristic is not None and self._stub_register_cfgs:
            for cn, (_b, c) in opt_requests.items():
                self._heuristic.register_cfg(cn, c)
        try:
            opt_outputs = self.rescale_bridge.evaluate_blocks(opt_requests)
        finally:
            if self._heuristic is not None and self._stub_register_cfgs:
                self._heuristic.clear_cfg_registry()

        opt_signals = aggregate_optimizer_signals(opt_outputs)
        any_invalid = bool(opt_signals.any_invalid)

        # 2) 把 effective rotations 反写到 cfg
        if not any_invalid:
            for cn, out in opt_outputs.items():
                try:
                    block_idx, _profile, layer_idx = parse_config_name(cn)
                except Exception:
                    continue
                if layer_idx < 0:
                    continue
                eff = out.raw.get("new_compact_config", {}).get("effective_rotations", [])
                if not eff:
                    continue
                # 从 rotation_name_map 拿到该 (block, profile) 的命名映射
                key = (int(block_idx), str(self.env_cfg.profile))
                name_map = (self.env_cfg.rotation_name_map or {}).get(key, {})
                flag_names = []
                for entry in eff:
                    src = entry if isinstance(entry, str) else entry.get("name")
                    if src is None:
                        continue
                    flag = name_map.get(str(src))
                    if flag:
                        flag_names.append(flag)
                if not flag_names:
                    continue
                # cfgs_dict[blockN][layer_idx] 上反写
                target_cfg = cfgs_dict[f"block{block_idx}"][int(layer_idx)]
                apply_rotation_flags_to_cfg(target_cfg, flag_names)

        # 3) invalid → 直接判死，不跑 forward
        info: Dict[str, Any] = {
            "decoded": decoded,
            "opt_signals": opt_signals,
            "opt_outputs_keys": list(opt_outputs.keys()),
            "invalid": any_invalid,
            "apply_failed": False,
            "eval_failed": False,
        }
        if degree_sync:
            info["model_degree_sync"] = degree_sync
        if any_invalid:
            metrics = EpisodeMetrics(
                loss_mean=float("inf"),
                loss_std=float("inf"),
                metric1_mean=0.0,
                metric2_mean=0.0,
            )
            breakdown = compute_reward(
                metrics, opt_signals,
                action_avg_k=avg_truncation_k_in_action(action_vec, self.num_layers),
                baseline=self.baseline,
                weights=self.reward_weights,
                acc_threshold=self.acc_threshold,
                stab_threshold=self.stab_threshold,
                any_invalid=True,
            )
            info["reward_breakdown"] = breakdown
            info["metrics"] = metrics
            self._step_idx += 1
            self._last_invalid_rate = 1.0
            self._last_total_bits_norm = float(opt_signals.total_bits_sum) / max(1.0, float(self.baseline.total_bits_sum))
            self._last_fusion_count = float(opt_signals.total_fusion_count)
            return self._build_state(), float(breakdown.reward), True, info

        # 4) 装 BLB 噪声
        try:
            self.bridge.apply(
                first_input_sf=int(decoded.first_input_sf),
                first_input_N=BLB_FIRST_INPUT_N,
                block1_cfgs=decoded.block1_cfgs,
                block2_cfgs=decoded.block2_cfgs,
                block3_cfgs=decoded.block3_cfgs,
                block4_cfgs=decoded.block4_cfgs,
                block5_cfgs=decoded.block5_cfgs,
            )
        except Exception as exc:
            try:
                self.bridge.clear()
            except Exception:
                pass
            # 互斥校验失败，按 invalid 处理
            metrics = EpisodeMetrics(loss_mean=float("inf"), loss_std=float("inf"))
            breakdown = compute_reward(
                metrics, opt_signals,
                action_avg_k=avg_truncation_k_in_action(action_vec, self.num_layers),
                baseline=self.baseline,
                weights=self.reward_weights,
                acc_threshold=self.acc_threshold,
                stab_threshold=self.stab_threshold,
                any_invalid=True,
            )
            info["reward_breakdown"] = breakdown
            info["error"] = f"BLB apply failed: {exc}"
            info["invalid"] = True
            info["apply_failed"] = True
            info["metrics"] = metrics
            self._step_idx += 1
            self._last_invalid_rate = 1.0
            self._last_total_bits_norm = float(opt_signals.total_bits_sum) / max(1.0, float(self.baseline.total_bits_sum))
            self._last_fusion_count = float(opt_signals.total_fusion_count)
            return self._build_state(), float(breakdown.reward), True, info

        # 5) forward + metrics（多 trial）
        try:
            metrics = self._eval_on_probe(self.env_cfg.num_trials_per_step)
        except Exception as exc:
            try:
                self.bridge.clear()
            except Exception:
                pass
            metrics = EpisodeMetrics(loss_mean=float("inf"), loss_std=float("inf"))
            breakdown = compute_reward(
                metrics, opt_signals,
                action_avg_k=avg_truncation_k_in_action(action_vec, self.num_layers),
                baseline=self.baseline,
                weights=self.reward_weights,
                acc_threshold=self.acc_threshold,
                stab_threshold=self.stab_threshold,
                any_invalid=True,
            )
            info["reward_breakdown"] = breakdown
            info["error"] = f"BLB eval failed: {exc}"
            info["invalid"] = True
            info["eval_failed"] = True
            info["metrics"] = metrics
            self._step_idx += 1
            self._last_invalid_rate = 1.0
            self._last_total_bits_norm = float(opt_signals.total_bits_sum) / max(1.0, float(self.baseline.total_bits_sum))
            self._last_fusion_count = float(opt_signals.total_fusion_count)
            return self._build_state(), float(breakdown.reward), True, info
        else:
            self.bridge.clear()

        # 6) reward
        breakdown = compute_reward(
            metrics, opt_signals,
            action_avg_k=avg_truncation_k_in_action(action_vec, self.num_layers),
            baseline=self.baseline,
            weights=self.reward_weights,
            acc_threshold=self.acc_threshold,
            stab_threshold=self.stab_threshold,
            any_invalid=False,
        )

        info["reward_breakdown"] = breakdown
        info["metrics"] = metrics

        # state 更新
        self._step_idx += 1
        self._last_invalid_rate = 0.0
        self._last_total_bits_norm = float(opt_signals.total_bits_sum) / max(1.0, float(self.baseline.total_bits_sum))
        self._last_fusion_count = float(opt_signals.total_fusion_count)

        return self._build_state(), float(breakdown.reward), True, info

    # ------------------------------------------------------------------
    # 评估子集 forward（单层 K-trials）
    # ------------------------------------------------------------------
    def _eval_on_probe(self, k_trials: int) -> EpisodeMetrics:
        """在 ``self.probe_batches`` 上跑 k_trials 次（独立 RNG），返回 EpisodeMetrics。"""
        k = max(1, int(k_trials))
        was_training = self.model.training
        self.model.eval()

        per_trial_loss: List[float] = []
        per_trial_metric1: List[float] = []
        per_trial_metric2: List[float] = []

        # 保存外层 RNG 状态以避免污染
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        np_rng = np.random.get_state()

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
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            np.random.set_state(np_rng)
            if was_training:
                self.model.train()

        if not per_trial_loss:
            return EpisodeMetrics()

        loss_arr = np.array(per_trial_loss, dtype=float)
        m1_arr = np.array(per_trial_metric1, dtype=float)
        m2_arr = np.array(per_trial_metric2, dtype=float)
        return EpisodeMetrics(
            loss_mean=float(loss_arr.mean()),
            loss_std=float(loss_arr.std(ddof=0)) if loss_arr.size > 1 else 0.0,
            metric1_mean=float(m1_arr.mean()),
            metric2_mean=float(m2_arr.mean()),
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
        ) -> BaselineCostStats:
    """在不装 BLB 的前提下，跑一遍"全 max-action"的 Rescale_optimizer，得到 baseline。

    spec §6.3 / §6.4：
      * 跑一次 baseline (全 max action) 拿 ``total_bits_sum`` / ``total_fusion_count``。
      * 跑若干 random action 估计典型 ``bits_drop`` / ``fusion_count`` / ``k_drop``，
        反推权重。
    """
    env.sync_degree_vectors_from_model()
    baseline_action = make_all_max_action_vector(env.num_layers)
    decoded = action_vector_to_cfgs(
        action_vec=baseline_action,
        max_sfs=env.max_sfs,
        num_layers=env.num_layers,
        gelu_degree=env.gelu_degree,
        attn_degree=env.attn_degree,
    )
    cfgs_dict = decoded.cfgs_dict()
    requests = build_optimizer_requests(env.env_cfg.profile, cfgs_dict)
    if env._heuristic is not None and env._stub_register_cfgs:
        for cn, (_b, c) in requests.items():
            env._heuristic.register_cfg(cn, c)
    try:
        outputs = env.rescale_bridge.evaluate_blocks(requests)
    finally:
        if env._heuristic is not None and env._stub_register_cfgs:
            env._heuristic.clear_cfg_registry()
    signals = aggregate_optimizer_signals(outputs)

    # 估计典型 drop
    bits_drops: List[float] = []
    fusion_counts: List[float] = []
    k_drops: List[float] = []
    rng = np.random.default_rng(seed=42)
    for _ in range(max(0, int(sample_count))):
        random_action = np.array(
            [rng.integers(0, d) for d in env.action_dims], dtype=int,
        )
        rd = action_vector_to_cfgs(
            random_action, env.max_sfs, env.num_layers,
            gelu_degree=env.gelu_degree, attn_degree=env.attn_degree,
        )
        rd_cfgs = rd.cfgs_dict()
        rd_requests = build_optimizer_requests(env.env_cfg.profile, rd_cfgs)
        if env._heuristic is not None and env._stub_register_cfgs:
            for cn, (_b, c) in rd_requests.items():
                env._heuristic.register_cfg(cn, c)
        try:
            rd_outputs = env.rescale_bridge.evaluate_blocks(rd_requests)
        finally:
            if env._heuristic is not None and env._stub_register_cfgs:
                env._heuristic.clear_cfg_registry()
        rd_signals = aggregate_optimizer_signals(rd_outputs)
        bits_drops.append(float(signals.total_bits_sum) - float(rd_signals.total_bits_sum))
        fusion_counts.append(float(rd_signals.total_fusion_count))
        avg_k = avg_truncation_k_in_action(random_action, env.num_layers)
        k_drops.append(float(max(K_LEVELS)) - float(avg_k))

    return BaselineCostStats(
        total_bits_sum=int(signals.total_bits_sum),
        total_fusion_count=int(signals.total_fusion_count),
        avg_k=float(max(K_LEVELS)),
        typical_bits_drop=float(np.mean(bits_drops)) if bits_drops else 1.0,
        typical_fusion_count=float(np.mean(fusion_counts)) if fusion_counts else 1.0,
        typical_k_drop=float(np.mean(k_drops)) if k_drops else 1.0,
    )
