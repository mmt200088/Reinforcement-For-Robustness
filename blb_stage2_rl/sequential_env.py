"""Per-block sequential RL env.

Wraps :class:`BLBStage2Env` to expose a horizon-N sequential interface where
each ``step()`` decides ONE (layer, block) pair instead of the whole 577-dim
action vector at once.

Decision schedule (L=12):
    step  0:  layer 0, block 2  (+ first_input fresh)
    step  1:  layer 0, block 3
    step  2:  layer 0, block 4
    step  3:  layer 0, block 5
    step  4:  layer 1, block 1
    step  5:  layer 1, block 2
    ...
    step 58:  layer 11, block 5      (terminal)

Reward: dense per-block ReplanSession-derived cost signal at every step,
plus the full base-env reward (accuracy + stability + total cost) at the
terminal step. Invalid chain at any per-block step yields a large penalty
and (optionally) early termination.

State at step t: static (degrees, num_layers) + step-position one-hot +
prev-step action history + prev-step optimizer signals (valid, total_bits,
fusion_count). The exact obs vector is built by :meth:`_build_obs`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from rescale_optimizer_bridge import (
    apply_optimizer_output_to_cfg,
    apply_rotation_flags_to_cfg,
    sync_block2_qk_binding,
    _strip_layer_suffix,
)

from .action_space import (
    BlockStepSpec,
    action_vector_to_cfgs,
    empty_full_action_vec,
    horizon_for_num_layers,
    splice_step_action_into_full_vec,
    step_schedule,
    step_schedule_max_dim,
)
from .env import BLBStage2Env


@dataclass
class SequentialEnvConfig:
    """Tunable knobs for the per-step reward shaping.

    All defaults are deliberately conservative: per-step shaping is much
    smaller than the terminal reward so the accuracy signal still dominates
    final-action ranking.
    """
    invalid_penalty: float = 1.0
    """Negative reward emitted on every step whose per-block ReplanSession
    returned invalid_chain != None. Tuned to be 5-10x larger than typical
    per-step cost shaping so the policy quickly learns to avoid invalid
    sub-actions."""

    cost_shaping_coeff: float = 0.05
    """Coefficient on the per-block (this_block_total_bits / baseline_block_total_bits)
    ratio. Subtracted from per-step reward so cheaper blocks earn more."""

    fusion_shaping_coeff: float = 0.0
    """Optional coefficient on per-block fusion_count. Default 0 (fusion is
    not directly meaningful at the per-block level; the full-eval terminal
    reward handles fusion in aggregate)."""

    early_terminate_on_invalid: bool = False
    """If True, the episode ends immediately when any per-block ReplanSession
    returns invalid_chain != None. Saves wasted forward-pass time but loses
    the chance to learn recovery from a bad early decision."""

    expose_prev_actions_in_obs: bool = True
    """If True, ``_build_obs`` includes a flat array of all previous per-step
    actions (padded to step_schedule_max_dim). If False, only step-position
    + summarised prev-optimizer signals are exposed."""

    expose_prev_optimizer_signals_in_obs: bool = True
    """If True, ``_build_obs`` includes per-prev-step (valid, total_bits_norm,
    fusion_count_norm) features."""


class BLBStage2SequentialEnv:
    """Sequential per-block RL env.

    Delegates the heavy lifting (model forward, full reward) to a wrapped
    :class:`BLBStage2Env`. Adds the horizon-N step machinery on top.

    Args:
        base_env:    A fully-constructed BLBStage2Env. Reused as-is for the
                     terminal full evaluation.
        env_cfg:     Per-step shaping knobs. Defaults are conservative.
    """

    def __init__(
            self,
            *,
            base_env: BLBStage2Env,
            env_cfg: Optional[SequentialEnvConfig] = None,
            ):
        self.base = base_env
        self.cfg = env_cfg or SequentialEnvConfig()
        self.num_layers = int(base_env.num_layers)
        self.profile = str(base_env.env_cfg.profile)
        self.horizon = horizon_for_num_layers(self.num_layers)
        self._max_step_dim = step_schedule_max_dim(self.num_layers)
        # Schedule depends on per-layer Stage-1 degrees; rebuild on reset to
        # pick up any model degree changes.
        self._schedule: List[BlockStepSpec] = []
        self._rebuild_schedule()

        # Per-episode mutable state
        self._pending_full_vec = empty_full_action_vec(self.num_layers)
        self._step_idx: int = 0
        self._prev_actions: List[List[int]] = []
        self._prev_signals: List[Dict[str, float]] = []
        self._prev_per_step_overrides: List[Dict[str, Any]] = []
        self._terminated_early: bool = False

    # ------------------------------------------------------------------
    # public surface
    # ------------------------------------------------------------------
    @property
    def schedule(self) -> List[BlockStepSpec]:
        return list(self._schedule)

    @property
    def max_step_dim(self) -> int:
        return self._max_step_dim

    @property
    def state_dim(self) -> int:
        """Width of the per-step observation vector."""
        return self._compute_obs_width()

    def step_spec(self, step_idx: int) -> BlockStepSpec:
        return self._schedule[int(step_idx)]

    def current_spec(self) -> BlockStepSpec:
        return self._schedule[self._step_idx]

    # ------------------------------------------------------------------
    # gym-like
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None) -> np.ndarray:
        # Delegate the heavy reset (handler restore, BLB clear, RNG seed,
        # degree sync) to the base env, then start a fresh accumulator.
        self.base.reset(seed=seed)
        self._rebuild_schedule()
        self._pending_full_vec = empty_full_action_vec(self.num_layers)
        self._step_idx = 0
        self._prev_actions = []
        self._prev_signals = []
        self._prev_per_step_overrides = []
        self._terminated_early = False
        return self._build_obs()

    def step(
            self,
            per_step_action: Sequence[int],
            ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Decide one (layer, block); return (obs, reward, done, info).

        ``per_step_action`` length must equal ``len(current_spec().slot_dims)``.
        """
        if self._terminated_early:
            raise RuntimeError("episode already terminated; call reset() before step()")

        spec = self.current_spec()
        action = np.asarray(per_step_action, dtype=int).reshape(-1)
        if action.size != len(spec.slot_dims):
            raise ValueError(
                f"step {spec.step_idx} expects {len(spec.slot_dims)} slots, got {action.size}"
            )

        # Splice into the accumulator so action_vector_to_cfgs sees it on the
        # terminal step (and so subsequent per-step decodes can use it as
        # context if the block depends on earlier blocks within the same layer).
        splice_step_action_into_full_vec(self._pending_full_vec, spec, action)

        # Decode this single (layer, block) cfg out of the partial accumulator.
        # action_vector_to_cfgs decodes ALL blocks for ALL layers; we pluck the
        # one we need. Slots not yet decided default to 0 -> baseline-low SF,
        # which is the safe lossless fallback for blocks that haven't reached
        # their step yet (they'll be re-decoded in the terminal step anyway).
        decoded = action_vector_to_cfgs(
            self._pending_full_vec,
            self.base.max_sfs,
            num_layers=self.num_layers,
            gelu_degree=self.base.gelu_degree,
            attn_degree=self.base.attn_degree,
        )
        cfgs_by_block = decoded.cfgs_dict()
        block_cfg = cfgs_by_block[f"block{spec.block_idx}"][spec.layer_idx]

        # Per-block ReplanSession call.
        config_name = f"{spec.graph_key_suffix}_L{spec.layer_idx}"
        try:
            out = self.base.rescale_bridge.evaluate(
                config_name=config_name,
                block_name=f"block{spec.block_idx}",
                cfg=block_cfg,
            )
        except Exception as exc:
            # If the bridge errors (missing graph_key etc.), surface a hard
            # invalid signal but don't crash the loop -- the runner can act
            # on it.
            self._record_step(spec, action, valid=False, total_bits=0,
                              fusion_count=0, error=str(exc))
            self._step_idx += 1
            self._terminated_early = True
            obs = self._build_obs()
            info = {
                "step": spec.step_idx, "block_idx": spec.block_idx, "layer_idx": spec.layer_idx,
                "graph_key": config_name, "valid": False,
                "bridge_error": str(exc), "early_terminated": True,
            }
            return obs, -float(self.cfg.invalid_penalty), True, info

        valid = bool(out.valid)
        total_bits = int(out.total_bits)
        fusion_count = int(out.fusion_count)
        invalid_chain = out.invalid_chain
        self._record_step(spec, action, valid=valid, total_bits=total_bits,
                          fusion_count=fusion_count)

        # Per-block reward shaping.
        per_step_reward = 0.0
        if not valid:
            per_step_reward -= float(self.cfg.invalid_penalty)
        else:
            # Normalise by this graph's baseline total_bits so the shaping
            # is comparable across blocks.
            graph_key = spec.graph_key_suffix
            baseline_total = self._lookup_baseline_block_total_bits(graph_key)
            if baseline_total and baseline_total > 0:
                cost_ratio = float(total_bits) / float(baseline_total)
                per_step_reward -= float(self.cfg.cost_shaping_coeff) * (cost_ratio - 1.0)
            if self.cfg.fusion_shaping_coeff != 0.0:
                per_step_reward -= float(self.cfg.fusion_shaping_coeff) * float(fusion_count)

        info: Dict[str, Any] = {
            "step": spec.step_idx,
            "block_idx": spec.block_idx,
            "layer_idx": spec.layer_idx,
            "graph_key": config_name,
            "valid": valid,
            "total_bits": total_bits,
            "fusion_count": fusion_count,
            "invalid_chain": invalid_chain,
        }

        # Optimizer-driven cfg override on the in-memory cfg, plus Q/K binding
        # sync (block 2 only). This mirrors what the single-shot env does after
        # evaluate_blocks, so the per-step decoded cfg ends up matching what
        # the optimizer settled on -- useful for downstream model install at
        # the terminal step.
        if valid:
            invoker_baselines: Mapping[str, Any] = (
                getattr(self.base.rescale_bridge.invoker, "baselines", {}) or {}
            )
            graph_key, _ = _strip_layer_suffix(config_name)
            baseline_entry = invoker_baselines.get(graph_key)
            baseline_skeleton = list(baseline_entry[0]) if baseline_entry else []
            rotation_name_map = (self.base.env_cfg.rotation_name_map or {}).get(
                (int(spec.block_idx), str(self.profile)), {}
            )
            overrides = apply_optimizer_output_to_cfg(
                block_cfg,
                output_raw=out.raw,
                block_idx=int(spec.block_idx),
                graph_key=graph_key,
                baseline_skeleton=baseline_skeleton,
                rotation_name_map=rotation_name_map,
            )
            if int(spec.block_idx) == 2:
                overrides = list(overrides) + sync_block2_qk_binding(block_cfg)
            if overrides:
                info["optimizer_cfg_overrides"] = [
                    (e.cfg_attr, e.source, e.old_value, e.new_value)
                    for e in overrides
                ]

        self._step_idx += 1
        done = (self._step_idx == self.horizon)
        if not valid and self.cfg.early_terminate_on_invalid:
            done = True
            self._terminated_early = True
            info["early_terminated"] = True

        if done and not self._terminated_early:
            # Terminal: hand off the accumulated full action vec to the base env
            # to run the actual model forward + full hard-priority reward.
            term_state, term_reward, _term_done, term_info = self.base.step(self._pending_full_vec)
            info["terminal_info"] = term_info
            info["terminal_reward"] = float(term_reward)
            return term_state, per_step_reward + float(term_reward), True, info

        # Non-terminal -> next obs
        obs = self._build_obs()
        return obs, per_step_reward, done, info

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _rebuild_schedule(self) -> None:
        def _broadcast(value, default: int) -> "list[int]":
            # base.{attn,gelu}_degree may be int, np.ndarray, or list; reshape
            # to a per-layer list of length num_layers. Single-value arrays
            # broadcast across layers.
            if isinstance(value, int):
                return [int(value)] * self.num_layers
            try:
                arr = list(np.asarray(value, dtype=int).reshape(-1).tolist())
            except Exception:
                return [int(default)] * self.num_layers
            if not arr:
                return [int(default)] * self.num_layers
            if len(arr) == 1:
                return [int(arr[0])] * self.num_layers
            if len(arr) != self.num_layers:
                # truncate or pad with last value to match expected length
                if len(arr) > self.num_layers:
                    arr = arr[: self.num_layers]
                else:
                    arr = arr + [arr[-1]] * (self.num_layers - len(arr))
            return [int(x) for x in arr]
        attn = _broadcast(self.base.attn_degree, 4)
        gelu = _broadcast(self.base.gelu_degree, 4)
        self._schedule = step_schedule(
            self.num_layers,
            profile=self.profile,
            attn_degree_per_layer=attn,
            gelu_degree_per_layer=gelu,
        )

    def _record_step(
            self,
            spec: BlockStepSpec,
            action: np.ndarray,
            *,
            valid: bool,
            total_bits: int,
            fusion_count: int,
            error: Optional[str] = None,
            ) -> None:
        # Pad the action to max_step_dim for fixed-shape obs concatenation.
        padded = list(action.tolist()) + [0] * (self._max_step_dim - len(action))
        self._prev_actions.append(padded[: self._max_step_dim])
        self._prev_signals.append({
            "valid": 1.0 if valid else 0.0,
            "total_bits": float(total_bits),
            "fusion_count": float(fusion_count),
            "error": 1.0 if error else 0.0,
        })

    def _lookup_baseline_block_total_bits(self, graph_key_suffix: str) -> Optional[float]:
        baselines = getattr(self.base.rescale_bridge.invoker, "baselines", None)
        if not isinstance(baselines, Mapping):
            return None
        entry = baselines.get(str(graph_key_suffix))
        if not entry:
            return None
        try:
            q_baseline = entry[2]
        except (IndexError, TypeError):
            return None
        if not q_baseline:
            return None
        # total_bits ~ sum of q-bits in the baseline modulus chain
        return float(sum(int(x) for x in q_baseline))

    # ----------------------------- obs encoding -----------------------------
    def _compute_obs_width(self) -> int:
        # Static (4) + current step one-hot over horizon
        # + current spec metadata (block_idx one-hot of 5, layer_idx normalised)
        # + optionally prev_actions (horizon * max_step_dim) + prev_signals
        # (horizon * 3).
        width = 4 + self.horizon + 5 + 1
        if self.cfg.expose_prev_actions_in_obs:
            width += self.horizon * self._max_step_dim
        if self.cfg.expose_prev_optimizer_signals_in_obs:
            width += self.horizon * 3
        return int(width)

    def _build_obs(self) -> np.ndarray:
        if self._step_idx >= self.horizon:
            # terminal -- caller should have stopped, but return a zeroed obs
            # so this never crashes.
            return np.zeros(self._compute_obs_width(), dtype=np.float32)

        spec = self.current_spec()
        parts: List[np.ndarray] = []

        # static
        parts.append(np.array([
            float(self.num_layers) / 24.0,
            float(self.base.attn_degree_state) / 8.0,
            float(self.base.gelu_degree_state) / 8.0,
            float(self.horizon) / 100.0,
        ], dtype=np.float32))

        # current step one-hot
        step_oh = np.zeros(self.horizon, dtype=np.float32)
        step_oh[self._step_idx] = 1.0
        parts.append(step_oh)

        # block_idx one-hot (1..5 -> indices 0..4) + layer_idx normalised
        blk_oh = np.zeros(5, dtype=np.float32)
        blk_oh[int(spec.block_idx) - 1] = 1.0
        parts.append(blk_oh)
        parts.append(np.array([float(spec.layer_idx) / max(1, self.num_layers - 1)], dtype=np.float32))

        # prev actions, padded to (horizon, max_step_dim) and filled with zeros
        # for steps we haven't reached yet
        if self.cfg.expose_prev_actions_in_obs:
            buf = np.zeros((self.horizon, self._max_step_dim), dtype=np.float32)
            for i, a in enumerate(self._prev_actions):
                buf[i, : len(a)] = np.asarray(a, dtype=np.float32) / 8.0  # rough normalisation
            parts.append(buf.reshape(-1))

        # prev signals: (valid, total_bits_norm, fusion_count_norm) per step
        if self.cfg.expose_prev_optimizer_signals_in_obs:
            buf = np.zeros((self.horizon, 3), dtype=np.float32)
            for i, s in enumerate(self._prev_signals):
                buf[i, 0] = float(s["valid"])
                buf[i, 1] = float(s["total_bits"]) / 1000.0
                buf[i, 2] = float(s["fusion_count"]) / 10.0
            parts.append(buf.reshape(-1))

        return np.concatenate(parts).astype(np.float32)
