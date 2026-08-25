"""BLB Stage 2 RL ``Env`` 包装。

不强依赖 ``gymnasium`` —— 我们只用最小化的 ``reset/step`` 接口，避免新增依赖。
"""
from __future__ import annotations

import copy
import contextlib
import hashlib
import math
import operator
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from blb_rl_bridge import BLBNoiseRLBridge
from rescale_optimizer_bridge import RescaleOptimizerBridge

from .action_space import (
    K_LEVELS,
    MaxSFsTable,
    action_dims_for_config,
    avg_truncation_k_in_action,
    make_all_max_action_vector,
    validate_action_vector,
)
from .candidate_store import action_hash
from .inference_eval import run_installed_probe_trial
from .optimizer_cost import materialize_action_for_model
from .probe_runner import (
    ProbeRunner,
    _normalize_probe_trial_result,
    diagnostics_payload,
)
from .reward import (
    BaselineCostStats,
    EpisodeMetrics,
    RewardBreakdown,
    RewardWeights,
    compute_reward,
)
from .statistical_constraints import TrialSeries, assess_candidate


_NULL_CTX = contextlib.nullcontext()


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
    profile: str = "mrpc"
    num_trials_per_step: int = 3


    borderline_retest_enabled: bool = True
    borderline_retest_trials_multiplier: int = 2
    probe_batch_count: int = 4
    deterministic_eval: bool = False
    rotation_name_map: Optional[Mapping[Tuple[int, str], Mapping[str, str]]] = None
    persistent_probe_install: bool = False
    truncation_backend: str = "binary"
    truncation_ring_bits: int = 43
    truncation_source_fractional_bits: int = 24
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


        self.acc_threshold_m2: Optional[float] = (
            float(acc_threshold_m2) if acc_threshold_m2 is not None else None
        )
        self.stab_threshold = float(stab_threshold)


        self.loss_threshold: Optional[float] = None
        self.statistical_reference = None
        self.statistical_gate_probability = 0.50
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


        self.probe_runner = probe_runner
        self._last_probe_diagnostics: Dict[str, Any] = {}
        self._installed_config_fingerprint: Optional[str] = None
        self.pareto_cost_archive = None


        self._probe_eval_counter: int = 0


        self.probe_noise_seed: Optional[int] = None


        self.probe_device_lock: Optional[Any] = None


        self.probe_device_lock_requires_sync: bool = False


        self.probe_noise_scope: Optional[str] = None


        self.probe_cuda_stream: Optional[Any] = None

        self.action_dims = action_dims_for_config(self.num_layers)
        self.total_action_dim = len(self.action_dims)


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
        self._installed_config_fingerprint = None

    def _materialize_action(
            self,
            action_vec: Sequence[int],
            *,
            boosted_overrides: Optional[Mapping[Tuple[int, int], Mapping[str, int]]] = None,
            ):
        """Run the one canonical action -> replan -> installable-cfg path."""
        bridge_invoker = getattr(self.rescale_bridge, "invoker", None)
        invoker_baselines: Mapping[str, Any] = (
            getattr(bridge_invoker, "baselines", {}) or {}
        )

        def rotation_provider(block_idx: int, profile: str) -> Mapping[str, str]:
            return (self.env_cfg.rotation_name_map or {}).get(
                (int(block_idx), str(profile)), {}
            )


        return materialize_action_for_model(
            action_vec,
            profile=self.env_cfg.profile,
            num_layers=self.num_layers,
            max_sfs=self.max_sfs,
            rescale_bridge=self.rescale_bridge,
            gelu_degree=self.gelu_degree,
            attn_degree=self.attn_degree,
            boosted_overrides=boosted_overrides,
            invoker_baselines=invoker_baselines,
            rotation_name_map_provider=rotation_provider,
            truncation_backend=self.env_cfg.truncation_backend,
            truncation_ring_bits=self.env_cfg.truncation_ring_bits,
            truncation_source_fractional_bits=(
                self.env_cfg.truncation_source_fractional_bits
            ),
            borrow_cached_optimizer_payloads=True,
        )

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


    @property
    def state_dim(self) -> int:

        return 6 + 4 + self.num_layers

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            ) -> np.ndarray:
        """清掉单表噪声和 BLB 残留，回到干净状态；返回 obs。"""

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
    def _normalize_trial_seeds(
            trial_seeds: Optional[Sequence[int]],
            trial_count: int,
            ) -> Tuple[int, ...]:
        if trial_seeds is None:
            return ()
        try:
            raw_trial_seeds = tuple(trial_seeds)
        except TypeError as exc:
            raise ValueError("trial_seeds must be an integer sequence") from exc
        if any(isinstance(seed, (bool, np.bool_)) for seed in raw_trial_seeds):
            raise ValueError("trial_seeds must be an integer sequence")
        try:
            normalized = tuple(operator.index(seed) for seed in raw_trial_seeds)
        except TypeError as exc:
            raise ValueError("trial_seeds must be an integer sequence") from exc
        if len(normalized) != int(trial_count):
            raise ValueError("trial_seeds length must match the number of probe trials")
        return normalized

    @classmethod
    def _trial_seeds_from_probe_diagnostics(
            cls,
            diagnostics: Any,
            trial_count: int,
            ) -> Tuple[int, ...]:
        seeds_by_trial: List[Optional[int]] = [None] * int(trial_count)
        for indices, seeds in zip(
                diagnostics.per_worker_trial_indices,
                diagnostics.per_worker_trial_seeds,
        ):
            if len(indices) != len(seeds):
                raise ValueError("probe diagnostics trial indices and seeds must align")
            for trial_idx, seed in zip(indices, seeds):
                index = operator.index(trial_idx)
                if not 0 <= index < int(trial_count) or seeds_by_trial[index] is not None:
                    raise ValueError("probe diagnostics trial indices must be unique and in range")
                seeds_by_trial[index] = seed
        if any(seed is None for seed in seeds_by_trial):
            raise ValueError("probe diagnostics must provide every trial seed")
        return cls._normalize_trial_seeds(seeds_by_trial, trial_count)

    @staticmethod
    def _metrics_from_trial_results(
            results: Sequence[Tuple[float, float, float]],
            *,
            trial_seeds: Optional[Sequence[int]] = None,
            ) -> EpisodeMetrics:
        normalized_trial_seeds = BLBStage2Env._normalize_trial_seeds(
            trial_seeds, len(results),
        ) if trial_seeds is not None else ()
        if not results:
            return EpisodeMetrics(trial_seeds=normalized_trial_seeds)
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
            loss_std=float(loss_arr.std(ddof=1)) if loss_arr.size > 1 else 0.0,
            metric1_mean=float(m1_arr.mean()),
            metric2_mean=float(m2_arr.mean()),
            metric1_std=float(m1_arr.std(ddof=1)) if m1_arr.size > 1 else 0.0,
            metric2_std=float(m2_arr.std(ddof=1)) if m2_arr.size > 1 else 0.0,
            loss_max=float(loss_arr.max()),
            metric1_min=float(m1_arr.min()),
            metric2_min=float(m2_arr.min()),
            loss_trials=tuple(float(value) for value in loss_arr.tolist()),
            metric1_trials=tuple(float(value) for value in m1_arr.tolist()),
            metric2_trials=tuple(float(value) for value in m2_arr.tolist()),
            trial_seeds=normalized_trial_seeds,
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

    def prepare_action_for_terminal_probe(
            self,
            action_vec: np.ndarray,
            *,
            external_cost_score: Optional[float] = None,
            external_cost_rank: Optional[float] = None,
            external_resource_objective: Optional[Mapping[str, Any]] = None,
            boosted_overrides: Optional[Mapping[Tuple[int, int], Mapping[str, int]]] = None,
            probe_base_seed: Optional[int] = None,
            ) -> Dict[str, Any]:
        """Prepare optimizer-adjusted cfgs for a terminal reward probe.

        This mirrors the pre-forward part of :meth:`step` and exists so the
        sequential trainer can batch several completed actions onto different
        GPUs before running model-forward reward probes.
        """
        action_vec = validate_action_vector(action_vec, self.num_layers)
        is_optimizer_baseline_action = bool(
            np.array_equal(action_vec, make_all_max_action_vector(self.num_layers))
        )
        action_vec_hash = action_hash(action_vec)
        degree_sync = self.sync_degree_vectors_from_model()
        timing: Dict[str, float] = {}
        cost_t0 = time.perf_counter()
        materialized = self._materialize_action(
            action_vec, boosted_overrides=boosted_overrides,
        )
        timing["cost_eval_wall_seconds"] = float(time.perf_counter() - cost_t0)
        decoded = materialized.decoded
        opt_outputs = materialized.outputs
        opt_signals = materialized.signals
        any_invalid = not bool(materialized.model_ready)
        optimizer_invalid_summary = (
            summarize_optimizer_invalid_outputs(opt_outputs)
            if materialized.optimizer_invalid else ""
        )
        replan_application = materialized.replan_application
        per_config_overrides = replan_application.get("optimizer_cfg_overrides", {})

        info: Dict[str, Any] = {
            "decoded": decoded,
            "opt_signals": opt_signals,
            "opt_outputs_keys": list(opt_outputs.keys()),
            "invalid": any_invalid,
            "apply_failed": False,
            "eval_failed": False,
            "forward_ran": False,
            "optimizer_baseline_action": bool(is_optimizer_baseline_action),
            "optimizer_eval_mode": materialized.optimizer_eval_mode,
            "materialization_failure_reason": materialized.failure_reason,
            "final_config_fingerprint": materialized.final_config_fingerprint,
            "timing": timing,
        }
        if optimizer_invalid_summary:
            info["optimizer_invalid_summary"] = optimizer_invalid_summary
        if degree_sync:
            info["model_degree_sync"] = degree_sync
        if per_config_overrides:
            info["optimizer_cfg_overrides"] = per_config_overrides
        info["replan_application"] = replan_application
        if not materialized.model_ready:
            info["forward_skipped_reason"] = str(materialized.failure_reason)

        prepared = {
            "action_vec": action_vec,
            "action_hash": action_vec_hash,
            "decoded": decoded,
            "opt_signals": opt_signals,
            "any_invalid": any_invalid,
            "requires_forward": bool(materialized.model_ready),
            "final_config_fingerprint": materialized.final_config_fingerprint,
            "info": info,
            "timing": timing,
        }
        if probe_base_seed is not None:
            prepared["probe_base_seed"] = int(probe_base_seed)
        if external_cost_score is not None:
            prepared["external_cost_score"] = float(external_cost_score)
        if external_cost_rank is not None:
            prepared["external_cost_rank"] = float(external_cost_rank)
        if external_resource_objective is not None:
            prepared["external_resource_objective"] = copy.deepcopy(
                dict(external_resource_objective)
            )
        return prepared

    def _compute_terminal_reward(
            self,
            metrics: EpisodeMetrics,
            opt_signals: Any,
            *,
            action_vec: np.ndarray,
            action_vec_hash: str,
            any_invalid: bool,
            external_cost_score: Optional[float],
            external_cost_rank: Optional[float],
            info: Dict[str, Any],
            external_resource_objective: Optional[Mapping[str, Any]] = None,
            ) -> RewardBreakdown:
        assessment = None
        if (
                str(getattr(self.reward_weights, "reward_design", "")).strip().lower()
                == "robust_constrained"
        ):
            if not any_invalid:
                reference = getattr(self, "statistical_reference", None)
                if reference is None:
                    raise RuntimeError(
                        "robust_constrained terminal reward requires statistical_reference"
                    )
                trials = TrialSeries(
                    loss=metrics.loss_trials,
                    metric1=metrics.metric1_trials,
                    metric2=metrics.metric2_trials,
                    seeds=metrics.trial_seeds,
                )
                seed_material = (
                    f"{action_vec_hash}:{int(reference.bootstrap_seed)}"
                ).encode("utf-8")
                bootstrap_seed = int.from_bytes(
                    hashlib.sha256(seed_material).digest()[:8], "big",
                ) & 0x7FFFFFFFFFFFFFFF
                assessment = assess_candidate(
                    trials,
                    reference,
                    gate_probability=float(getattr(
                        self, "statistical_gate_probability", 0.50,
                    )),
                    bootstrap_seed=bootstrap_seed,
                )
                info["statistical_assessment_provenance"] = "exact_trials"
                info["statistical_assessment"] = {
                    **asdict(assessment),
                    "bootstrap_seed": int(bootstrap_seed),
                }
                info["statistical_trials"] = {
                    "loss": [float(value) for value in trials.loss],
                    "metric1": [float(value) for value in trials.metric1],
                    "metric2": [float(value) for value in trials.metric2],
                    "seeds": [int(value) for value in trials.seeds],
                }

        return compute_reward(
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
            external_resource_objective=external_resource_objective,
            loss_threshold=self.loss_threshold,
            constraint_assessment=assessment,
        )

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

        breakdown = self._compute_terminal_reward(
            metrics,
            opt_signals,
            action_vec=action_vec,
            action_vec_hash=action_vec_hash,
            any_invalid=any_invalid,
            external_cost_score=prepared.get("external_cost_score"),
            external_cost_rank=prepared.get("external_cost_rank"),
            external_resource_objective=prepared.get("external_resource_objective"),
            info=info,
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
        grouped_probe_enabled = bool(
            forward_indices
            and self.probe_runner is not None
            and hasattr(self.probe_runner, "run_action_trial_groups")
            and all(
                item.get("probe_base_seed") is not None
                for item in prepared_items
                if bool(item.get("requires_forward", True))
            )
        )
        if grouped_probe_enabled:
            from .seed_utils import derive_probe_trial_seed

            decoded_actions = [
                prepared_items[index]["decoded"] for index in forward_indices
            ]
            base_seeds = [
                int(prepared_items[index]["probe_base_seed"])
                for index in forward_indices
            ]
            try:
                grouped_results = self.probe_runner.run_action_trial_groups(
                    decoded_actions,
                    base_seeds=base_seeds,
                    k=k,
                )
            except Exception as exc:
                raise RuntimeError(
                    "exact grouped terminal probe failed; refusing to "
                    "change per-episode failure semantics"
                ) from exc
            self._probe_eval_counter += len(forward_indices)
            self._installed_config_fingerprint = None
            diag_obj = self.probe_runner.last_diagnostics
            group_diag = diagnostics_payload(diag_obj) if diag_obj is not None else {}
            group_wall = float(group_diag.get("wall_seconds", 0.0) or 0.0)
            action_count = len(forward_indices)
            amortized_wall = group_wall / max(1, action_count)
            group_worker_seconds = [
                float(value)
                for value in group_diag.get("per_worker_seconds", ())
            ]
            group_metadata = {
                "group_k": int(group_diag.get("k", action_count * k)),
                "group_action_count": int(
                    group_diag.get("action_count", action_count)
                ),
                "group_trials_per_action": int(
                    group_diag.get("trials_per_action", k)
                ),
                "group_per_worker_seconds": list(group_worker_seconds),
                "group_per_worker_trial_counts": list(
                    group_diag.get("per_worker_trial_counts", ())
                ),
                "group_per_worker_trial_indices": copy.deepcopy(
                    group_diag.get("per_worker_trial_indices", ())
                ),
                "group_per_worker_trial_seeds": copy.deepcopy(
                    group_diag.get("per_worker_trial_seeds", ())
                ),
                "group_per_worker_action_trial_indices": copy.deepcopy(
                    group_diag.get("per_worker_action_trial_indices", ())
                ),
                "group_line": str(group_diag.get("line", "")),
                "group_multi_action": bool(
                    group_diag.get("multi_action", False)
                ),
            }
            grouped_by_item = {
                item_index: (local_index, grouped_results[local_index])
                for local_index, item_index in enumerate(forward_indices)
            }
            for item_index, item in enumerate(prepared_items):
                if item_index not in grouped_by_item:
                    metrics = self._placeholder_metrics_for_invalid()
                    out[item_index] = self._finish_prepared_terminal_probe(
                        item,
                        metrics,
                        probe_diagnostics={
                            "fast_reward_mode": True,
                            "validation_required": bool(validation_required),
                        },
                        forward_ran=False,
                    )
                    continue
                local_index, trial_results = grouped_by_item[item_index]
                base_seed = base_seeds[local_index]
                trial_seeds = [
                    derive_probe_trial_seed(base_seed, trial_index)
                    for trial_index in range(k)
                ]
                per_worker_indices = []
                for tasks in getattr(
                        diag_obj, "per_worker_action_trial_indices", (),
                ):
                    per_worker_indices.append([
                        int(trial_index)
                        for action_index, trial_index in tasks
                        if int(action_index) == int(local_index)
                    ])
                per_worker_seeds = [
                    [
                        derive_probe_trial_seed(base_seed, trial_index)
                        for trial_index in indices
                    ]
                    for indices in per_worker_indices
                ]
                diag = {
                    **group_diag,
                    **group_metadata,
                    "k": int(k),
                    "action_count": 1,
                    "trials_per_action": int(k),
                    "multi_action": False,
                    "wall_seconds": float(amortized_wall),
                    "group_wall_seconds": float(group_wall),
                    "per_worker_seconds": [
                        value / max(1, action_count)
                        for value in group_worker_seconds
                    ],
                    "per_worker_trial_counts": [
                        len(indices) for indices in per_worker_indices
                    ],
                    "per_worker_trial_indices": per_worker_indices,
                    "per_worker_trial_seeds": per_worker_seeds,
                    "per_worker_action_trial_indices": [
                        [[0, int(trial_index)] for trial_index in indices]
                        for indices in per_worker_indices
                    ],
                    "line": (
                        "[probe-runner] grouped action="
                        f"{int(local_index)} k={int(k)} "
                        f"group_actions={int(action_count)} "
                        f"group_wall={float(group_wall):.3f}s"
                    ),
                    "fast_reward_mode": True,
                    "online_num_trials_per_step": int(k),
                    "terminal_eval_batch_size": int(action_count),
                    "validation_required": bool(validation_required),
                    "probe_install_wall_seconds": float(amortized_wall),
                    "probe_install_skipped": False,
                    "probe_clear_wall_seconds": 0.0,
                    "probe_clear_skipped": True,
                    "persistent_probe_install": True,
                }
                metrics = self._metrics_from_trial_results(
                    trial_results,
                    trial_seeds=trial_seeds,
                )
                self._last_probe_diagnostics = dict(diag)
                out[item_index] = self._finish_prepared_terminal_probe(
                    item,
                    metrics,
                    probe_diagnostics=diag,
                    forward_ran=True,
                )
            return [result for result in out if result is not None]

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
                local_trial_seeds: Optional[Tuple[int, ...]] = None
                if diag_obj is not None:
                    local_trial_seeds = self._trial_seeds_from_probe_diagnostics(
                        diag_obj, len(forward_indices),
                    )
                    diag = diagnostics_payload(diag_obj)
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
                    metrics = self._metrics_from_trial_results(
                        [result],
                        trial_seeds=(
                            [local_trial_seeds[local_idx]]
                            if local_trial_seeds is not None else None
                        ),
                    )
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
                config_fingerprint = str(item["final_config_fingerprint"])
                persistent_install = bool(getattr(self.env_cfg, "persistent_probe_install", False))
                install_skipped = False
                install_t0 = time.perf_counter()
                try:
                    if (
                            persistent_install
                            and self._installed_config_fingerprint == config_fingerprint
                    ):
                        install_skipped = True
                    elif self.probe_runner is not None:
                        self.probe_runner.install_action(decoded)
                        self._installed_config_fingerprint = (
                            config_fingerprint if persistent_install else None
                        )
                    else:
                        self.bridge.apply(
                            block1_cfgs=decoded.block1_cfgs,
                            block2_cfgs=decoded.block2_cfgs,
                            block3_cfgs=decoded.block3_cfgs,
                            block4_cfgs=decoded.block4_cfgs,
                            block5_cfgs=decoded.block5_cfgs,
                        )
                        self._installed_config_fingerprint = (
                            config_fingerprint if persistent_install else None
                        )
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
            external_resource_objective: Optional[Mapping[str, Any]] = None,
            boosted_overrides: Optional[Mapping[Tuple[int, int], Mapping[str, int]]] = None,
            ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """单步 episode：装噪声 → forward → 计算 reward → 还原噪声。

        ``external_cost_score`` / ``external_cost_rank``：fusion-count 路径专用。
        sequential_env 终局把 per-block 加权 cost 节省算好传进来，只在最终 valid
        reward（P3）里替掉聚合 fusion/K/bits cost。非 fusion 调用保持 None。
        invalid 分支必为 P1，compute_reward 不读 cost，故无需透传。

        ``boosted_overrides``：加大精度专用。``{(block_idx, layer_idx): {field: sf}}``
        —— 选中的 boosted fusion option 的显式 SF（含选定 K）。因为 ``action_vec`` 只能
        携带网格动作索引、表达不了高于 baseline 的 boosted SF，这里把对应 (block, layer)
        的 cfg 用 SF-direct 重建，使 **本次 forward 真正安装的噪声是加大精度之后的动作组**
        （cost replan / optimizer override / 装噪声 全部基于 boosted cfg）。
        """
        action_vec = validate_action_vector(action_vec, self.num_layers)
        is_optimizer_baseline_action = bool(
            np.array_equal(action_vec, make_all_max_action_vector(self.num_layers))
        )
        action_vec_hash = action_hash(action_vec)

        degree_sync = self.sync_degree_vectors_from_model()
        timing: Dict[str, float] = {}
        cost_t0 = time.perf_counter()
        materialized = self._materialize_action(
            action_vec, boosted_overrides=boosted_overrides,
        )
        timing["cost_eval_wall_seconds"] = float(time.perf_counter() - cost_t0)
        decoded = materialized.decoded


        opt_outputs = materialized.outputs
        opt_signals = materialized.signals
        any_invalid = not bool(materialized.model_ready)
        optimizer_invalid_summary = (
            summarize_optimizer_invalid_outputs(opt_outputs)
            if materialized.optimizer_invalid else ""
        )
        replan_application = materialized.replan_application
        per_config_overrides = replan_application.get("optimizer_cfg_overrides", {})


        info: Dict[str, Any] = {
            "decoded": decoded,
            "opt_signals": opt_signals,
            "opt_outputs_keys": list(opt_outputs.keys()),
            "invalid": any_invalid,
            "apply_failed": False,
            "eval_failed": False,
            "forward_ran": False,
            "optimizer_baseline_action": bool(is_optimizer_baseline_action),
            "optimizer_eval_mode": materialized.optimizer_eval_mode,
            "materialization_failure_reason": materialized.failure_reason,
            "final_config_fingerprint": materialized.final_config_fingerprint,
            "timing": timing,
        }
        if optimizer_invalid_summary:
            info["optimizer_invalid_summary"] = optimizer_invalid_summary
        if degree_sync:
            info["model_degree_sync"] = degree_sync
        if per_config_overrides:
            info["optimizer_cfg_overrides"] = per_config_overrides
        info["replan_application"] = replan_application


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
            breakdown = self._compute_terminal_reward(
                metrics,
                opt_signals,
                action_vec=action_vec,
                action_vec_hash=action_vec_hash,
                any_invalid=True,
                external_cost_score=external_cost_score,
                external_cost_rank=external_cost_rank,
                external_resource_objective=external_resource_objective,
                info=info,
            )
            info["reward_breakdown"] = breakdown
            info["action_hash"] = action_vec_hash
            info["metrics"] = metrics
            info["forward_ran"] = False
            info["forward_skipped_reason"] = str(materialized.failure_reason)
            info["invalid"] = True
            self._step_idx += 1
            self._last_invalid_rate = 1.0
            self._last_total_bits_norm = float(opt_signals.total_bits_sum) / max(
                1.0, float(self.baseline.total_bits_sum)
            )
            self._last_fusion_count = float(opt_signals.total_fusion_count)
            return self._build_state(), float(breakdown.reward), True, info


        persistent_install = bool(getattr(self.env_cfg, "persistent_probe_install", False))
        config_fingerprint = materialized.final_config_fingerprint
        install_skipped = False
        install_t0 = time.perf_counter()
        try:
            if (
                    persistent_install
                    and self._installed_config_fingerprint == config_fingerprint
            ):
                install_skipped = True
            elif self.probe_runner is not None:
                self.probe_runner.install_action(decoded)
                self._installed_config_fingerprint = (
                    config_fingerprint if persistent_install else None
                )
            else:
                self.bridge.apply(
                    block1_cfgs=decoded.block1_cfgs,
                    block2_cfgs=decoded.block2_cfgs,
                    block3_cfgs=decoded.block3_cfgs,
                    block4_cfgs=decoded.block4_cfgs,
                    block5_cfgs=decoded.block5_cfgs,
                )
                self._installed_config_fingerprint = (
                    config_fingerprint if persistent_install else None
                )
        except Exception as exc:
            try:
                self.clear_installed_blb()
            except Exception:
                pass
            timing["probe_install_wall_seconds"] = float(time.perf_counter() - install_t0)
            timing["probe_install_skipped"] = float(0.0)

            metrics = EpisodeMetrics(loss_mean=float("inf"), loss_std=float("inf"))
            breakdown = self._compute_terminal_reward(
                metrics,
                opt_signals,
                action_vec=action_vec,
                action_vec_hash=action_vec_hash,
                any_invalid=True,
                external_cost_score=external_cost_score,
                external_cost_rank=external_cost_rank,
                external_resource_objective=external_resource_objective,
                info=info,
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
            breakdown = self._compute_terminal_reward(
                metrics,
                opt_signals,
                action_vec=action_vec,
                action_vec_hash=action_vec_hash,
                any_invalid=True,
                external_cost_score=external_cost_score,
                external_cost_rank=external_cost_rank,
                external_resource_objective=external_resource_objective,
                info=info,
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


        breakdown = self._compute_terminal_reward(
            metrics,
            opt_signals,
            action_vec=action_vec,
            action_vec_hash=action_vec_hash,
            any_invalid=any_invalid,
            external_cost_score=external_cost_score,
            external_cost_rank=external_cost_rank,
            external_resource_objective=external_resource_objective,
            info=info,
        )

        info["reward_breakdown"] = breakdown
        info["action_hash"] = action_vec_hash
        info["metrics"] = metrics


        self._step_idx += 1
        self._last_invalid_rate = 1.0 if any_invalid else 0.0
        self._last_total_bits_norm = float(opt_signals.total_bits_sum) / max(1.0, float(self.baseline.total_bits_sum))
        self._last_fusion_count = float(opt_signals.total_fusion_count)

        return self._build_state(), float(breakdown.reward), True, info


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
        near-miss band, on the direct random-probe path, or when disabled.
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


        if self.probe_noise_seed is not None and self.probe_runner is None:
            return self._eval_on_probe_deterministic(k)


        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        np_rng = np.random.get_state()

        per_trial_loss: List[float] = []
        per_trial_metric1: List[float] = []
        per_trial_metric2: List[float] = []
        trial_seeds: Optional[List[int]] = None
        probe_wall_start = time.perf_counter()

        try:
            if self.probe_runner is not None:

                base_seed = (
                    int(self.probe_noise_seed)
                    if self.probe_noise_seed is not None
                    else self._derive_probe_base_seed()
                )
                self._probe_eval_counter += 1
                results = self.probe_runner.run_trials(k, base_seed=base_seed)
                diag = self.probe_runner.last_diagnostics
                if diag is not None:
                    self._last_probe_diagnostics = diagnostics_payload(diag)
                    trial_seeds = list(
                        self._trial_seeds_from_probe_diagnostics(diag, k)
                    )
                for (loss, m1, m2) in results:
                    if loss is None or (isinstance(loss, float) and not math.isfinite(loss)):


                        per_trial_loss.append(float(loss) if loss is not None else float("nan"))
                    else:
                        per_trial_loss.append(float(loss))
                    per_trial_metric1.append(float(m1))
                    per_trial_metric2.append(float(m2))
            else:

                was_training = self.model.training
                self.model.eval()
                try:
                    for trial_idx in range(k):

                        seed = (int(time.time_ns()) ^ (trial_idx * 1_000_003)) & 0x7FFFFFFFFFFFFFFF
                        torch.manual_seed(seed)
                        np.random.seed(seed % (2**32))
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(seed)

                        loss, m1, m2 = run_installed_probe_trial(
                            self.model,
                            self.probe_batches,
                            is_regression=bool(self.is_regression),
                            metric_profile=str(
                                getattr(getattr(self, "env_cfg", None), "profile", "") or ""
                            ),
                            restore_training=False,
                        )
                        per_trial_loss.append(loss)
                        per_trial_metric1.append(m1)
                        per_trial_metric2.append(m2)
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
            trial_seeds=trial_seeds,
        )

    def _run_deterministic_probe_trial_indices(
            self,
            trial_indices: Sequence[int],
            ) -> tuple[List[Tuple[float, float, float]], List[int]]:
        from function_handler import noise_rng_scope, reseed_noise_rng_for_device
        from .seed_utils import derive_probe_trial_seed

        indices = tuple(int(trial_index) for trial_index in trial_indices)
        if any(trial_index < 0 for trial_index in indices):
            raise ValueError("deterministic probe trial indices must be nonnegative")
        if len(set(indices)) != len(indices):
            raise ValueError("deterministic probe trial indices must be unique")
        scope = getattr(self, "probe_noise_scope", None)
        lock = self.probe_device_lock
        if scope is not None:
            lock = _NULL_CTX
        if lock is None:
            lock = _NULL_CTX
        sync_before_unlock = (
            bool(getattr(self, "probe_device_lock_requires_sync", False))
            and getattr(self._device, "type", None) == "cuda"
            and scope is None
        )
        probe_stream = getattr(self, "probe_cuda_stream", None)
        use_probe_stream = (
            probe_stream is not None
            and getattr(self._device, "type", None) == "cuda"
        )
        base_seed = int(self.probe_noise_seed)
        trial_seeds = [
            derive_probe_trial_seed(base_seed, trial_index)
            for trial_index in indices
        ]
        results: List[Tuple[float, float, float]] = []
        probe_wall_start = time.perf_counter()
        was_training = self.model.training
        if was_training:
            self.model.eval()

        @contextlib.contextmanager
        def _forward_context(seed: int):
            with noise_rng_scope(scope):
                with lock:
                    reseed_noise_rng_for_device(self._device, seed)
                    stream_ctx = (
                        torch.cuda.stream(probe_stream)
                        if use_probe_stream else _NULL_CTX
                    )
                    with stream_ctx:
                        yield
                    if use_probe_stream:
                        torch.cuda.current_stream(self._device).wait_stream(
                            probe_stream
                        )
                    if sync_before_unlock:
                        torch.cuda.synchronize(self._device)

        try:
            for position, _trial_index in enumerate(indices):
                seed = trial_seeds[position]
                result = run_installed_probe_trial(
                    self.model,
                    self.probe_batches,
                    is_regression=bool(self.is_regression),
                    metric_profile=str(
                        getattr(
                            getattr(self, "env_cfg", None),
                            "profile", "",
                        ) or ""
                    ),
                    restore_training=False,
                    forward_context=_forward_context(seed),
                )
                results.append(_normalize_probe_trial_result(result))
        finally:
            if was_training:
                self.model.train()
        wall_elapsed = time.perf_counter() - probe_wall_start
        self._last_probe_diagnostics = {
            "k": len(indices),
            "wall_seconds": float(wall_elapsed),
            "per_worker_seconds": [float(wall_elapsed)],
            "per_worker_trial_counts": [len(indices)],
            "per_worker_trial_indices": [list(indices)],
            "per_worker_trial_seeds": [list(trial_seeds)],
            "devices": [str(self._device)],
            "speedup_vs_sequential": 1.0,
            "deterministic_probe_seed": int(base_seed),
            "line": (
                f"[probe-deterministic] k={len(indices)} device={self._device} "
                f"base_seed={base_seed} wall={wall_elapsed:.3f}s"
            ),
        }
        return results, trial_seeds

    def _eval_on_probe_deterministic(self, k: int) -> EpisodeMetrics:
        """K serial trials on this env's device with keyed noise seeds."""
        results, trial_seeds = self._run_deterministic_probe_trial_indices(
            range(int(k))
        )
        return self._aggregate_probe_trials(
            [result[0] for result in results],
            [result[1] for result in results],
            [result[2] for result in results],
            trial_seeds=trial_seeds,
        )

    def _aggregate_probe_trials(
            self,
            per_trial_loss: List[float],
            per_trial_metric1: List[float],
            per_trial_metric2: List[float],
            trial_seeds: Optional[Sequence[int]] = None,
            ) -> EpisodeMetrics:
        trial_count = len(per_trial_loss)
        if len(per_trial_metric1) != trial_count or len(per_trial_metric2) != trial_count:
            raise ValueError("probe trial channels must have equal lengths")
        normalized_trial_seeds = BLBStage2Env._normalize_trial_seeds(
            trial_seeds, trial_count,
        )
        if not per_trial_loss:
            return EpisodeMetrics(trial_seeds=normalized_trial_seeds)

        loss_arr = np.array(per_trial_loss, dtype=float)
        m1_arr = np.array(per_trial_metric1, dtype=float)
        m2_arr = np.array(per_trial_metric2, dtype=float)


        _LOSS_CAP = 100.0
        loss_arr = np.nan_to_num(loss_arr, nan=_LOSS_CAP, posinf=_LOSS_CAP, neginf=_LOSS_CAP)
        loss_arr = np.clip(loss_arr, 0.0, _LOSS_CAP)
        m1_arr = np.nan_to_num(m1_arr, nan=0.0, posinf=1.0, neginf=0.0)
        m2_arr = np.nan_to_num(m2_arr, nan=0.0, posinf=1.0, neginf=0.0)

        return EpisodeMetrics(
            loss_mean=float(loss_arr.mean()),
            loss_std=float(loss_arr.std(ddof=1)) if loss_arr.size > 1 else 0.0,
            metric1_mean=float(m1_arr.mean()),
            metric2_mean=float(m2_arr.mean()),
            metric1_std=float(m1_arr.std(ddof=1)) if m1_arr.size > 1 else 0.0,
            metric2_std=float(m2_arr.std(ddof=1)) if m2_arr.size > 1 else 0.0,
            loss_max=float(loss_arr.max()),
            metric1_min=float(m1_arr.min()),
            metric2_min=float(m2_arr.min()),
            loss_trials=tuple(float(value) for value in loss_arr.tolist()),
            metric1_trials=tuple(float(value) for value in m1_arr.tolist()),
            metric2_trials=tuple(float(value) for value in m2_arr.tolist()),
            trial_seeds=normalized_trial_seeds,
        )


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

        target_dim = self.state_dim
        if state.shape[0] < target_dim:
            pad = np.zeros(target_dim - state.shape[0], dtype=np.float32)
            state = np.concatenate([state, pad], axis=0)
        elif state.shape[0] > target_dim:
            state = state[:target_dim]
        return state


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


    bits_drops: List[float] = []
    fusion_counts: List[float] = []
    k_drops: List[float] = []
    rng = np.random.default_rng(seed=42)
    for _ in range(max(0, int(sample_count))):
        random_action = np.array(
            [rng.integers(0, d) for d in env.action_dims], dtype=int,
        )
        materialized = env._materialize_action(random_action)
        rd_signals = materialized.signals
        if not materialized.model_ready:
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
