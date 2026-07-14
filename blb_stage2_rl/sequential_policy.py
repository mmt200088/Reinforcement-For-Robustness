"""Sequential PPO actor-critic + rollout buffer for the per-block env.

Three pieces:
  1. :class:`BLBStage2SequentialPolicy` -- shared trunk + vectorized per-slot
     heads sized to ``step_schedule_max_dim`` (24 for the current L=12 action
     space).
     At each step the env tells the policy which slots are *active* (a
     boolean mask of length max_step_dim, padding suppressed via -inf logits)
     and what the per-slot ``num_levels`` are (so logits beyond that get
     masked out as well).

  2. :class:`SequentialRolloutBuffer` -- stores per-step
     (state, action, log_prob, value, reward, done, slot_mask,
     action_level_mask) tuples organised by episode. ``compute_gae(...)``
     produces λ-returns and advantages over the horizon.

  3. :func:`sequential_ppo_update` -- standard PPO-clip update over the
     buffered transitions. Aware of variable per-step action width via
     the slot_mask.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _SlotHeadView:
    """Read-only compatibility view for legacy tests/introspection.

    The real actor head is vectorized in ``slot_head_weight`` /
    ``slot_head_bias``; exposing lightweight ``weight`` and ``bias`` slices
    keeps old initialization checks meaningful without putting a Python
    ``ModuleList`` back on the hot path.
    """
    weight: torch.Tensor
    bias: torch.Tensor

@dataclass
class SequentialPolicyConfig:
    """Hyper-params for the sequential actor-critic.

    The head outputs ``max_step_dim * max_num_levels`` logits. At sample /
    evaluate time we pick the first ``len(active_slots)`` rows and within each
    row the first ``num_levels[k]`` columns. Padding is implemented with
    -inf masks so the categorical never assigns probability to invalid slots.
    """
    state_dim: int
    max_step_dim: int
    max_num_levels: int = 6      # max levels across F/W/M/S/R/K (currently 6 for K)
    d_hidden: int = 256
    d_step_embed: int = 32       # embedding for step_idx (0..horizon-1)
    horizon: int = 59
    block_count: int = 5         # block one-hot dim (1..5)
    num_layers: int = 12         # for layer one-hot if needed by the trunk
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    step_embed_dim: int = 16
    layer_embed_dim: int = 16
    block_embed_dim: int = 8
    prev_action_embed_dim: int = 4
    cont_proj_dim: int = 64
    actor_dim: int = 64
    critic_dim: int = 64
    default_prior_scale: float = 0.0
    metadata_width: int = 6
    signal_width: int = 3
    step_layer_indices: Optional[Sequence[int]] = None
    step_block_indices: Optional[Sequence[int]] = None

    def __post_init__(self) -> None:
        if (self.step_layer_indices is None) != (self.step_block_indices is None):
            raise ValueError(
                "step_layer_indices and step_block_indices must be supplied together"
            )
        if int(self.metadata_width) < 0:
            raise ValueError("metadata_width must be nonnegative")
        if int(self.signal_width) <= 0:
            raise ValueError("signal_width must be positive")
        self.metadata_width = int(self.metadata_width)
        self.signal_width = int(self.signal_width)
        if self.step_layer_indices is None:
            return
        layer_indices = tuple(int(value) for value in self.step_layer_indices)
        block_indices = tuple(int(value) for value in self.step_block_indices)
        if len(layer_indices) != int(self.horizon):
            raise ValueError(
                f"step_layer_indices has {len(layer_indices)} values, "
                f"expected horizon={self.horizon}"
            )
        if len(block_indices) != int(self.horizon):
            raise ValueError(
                f"step_block_indices has {len(block_indices)} values, "
                f"expected horizon={self.horizon}"
            )
        if any(value < 0 or value >= int(self.num_layers) for value in layer_indices):
            raise ValueError(
                f"step_layer_indices values must be in [0, {self.num_layers})"
            )
        if any(value < 0 or value >= int(self.block_count) for value in block_indices):
            raise ValueError(
                f"step_block_indices values must be in [0, {self.block_count})"
            )
        self.step_layer_indices = layer_indices
        self.step_block_indices = block_indices


class RunningMeanStd:
    """Tiny Welford running statistics helper for return normalization."""

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = float(epsilon)

    @property
    def std(self) -> float:
        return float(math.sqrt(float(self.var) + 1e-8))

    def update(self, values: Any) -> None:
        if isinstance(values, torch.Tensor):
            values_detached = values.detach()
            batch_count = int(values_detached.numel())
            if batch_count == 0:
                return
            values_stats = values_detached.reshape(-1).to(dtype=torch.float64)
            stats = torch.stack((
                values_stats.mean(),
                values_stats.var(unbiased=False),
            ))
            batch_mean, batch_var = (float(x) for x in stats.detach().cpu().numpy())
            self._update_from_moments(batch_mean, batch_var, batch_count)
            return
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return
        batch_mean = float(np.mean(arr))
        batch_var = float(np.var(arr))
        batch_count = int(arr.size)
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
            self,
            batch_mean: float,
            batch_var: float,
            batch_count: int,
            ) -> None:
        if int(batch_count) <= 0:
            return
        delta = float(batch_mean) - float(self.mean)
        total_count = float(self.count) + float(batch_count)
        self.mean = float(self.mean) + delta * float(batch_count) / total_count
        m_a = float(self.var) * float(self.count)
        m_b = float(batch_var) * float(batch_count)
        m2 = m_a + m_b + delta * delta * float(self.count) * float(batch_count) / total_count
        self.var = float(m2 / total_count)
        self.count = float(total_count)

    def normalize(self, values: Any) -> Any:
        if isinstance(values, torch.Tensor):
            return (values - float(self.mean)) / (float(self.std) + 1e-8)
        return (values - float(self.mean)) / (float(self.std) + 1e-8)

    def state_dict(self) -> Dict[str, float]:
        return {
            "mean": float(self.mean),
            "var": float(self.var),
            "count": float(self.count),
        }

    def load_state_dict(self, state: Optional[Dict[str, Any]]) -> None:
        if not isinstance(state, dict):
            return
        self.mean = float(state.get("mean", self.mean))
        self.var = float(state.get("var", self.var))
        self.count = float(state.get("count", self.count))


class GRUGate(nn.Module):
    """GTrXL-style gated residual path."""

    def __init__(self, d_model: int):
        super().__init__()
        self.linear_r = nn.Linear(d_model * 2, d_model)
        self.linear_z = nn.Linear(d_model * 2, d_model)
        self.linear_h = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([x, y], dim=-1)
        r = torch.sigmoid(self.linear_r(concat))
        z = torch.sigmoid(self.linear_z(concat))
        h = torch.tanh(self.linear_h(torch.cat([x, r * y], dim=-1)))
        return (1.0 - z) * x + z * h


class SequentialGTrXLBlock(nn.Module):
    """Pre-LN causal self-attention + gated residual FFN block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.gate1 = GRUGate(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(d_ff, d_model),
            nn.Dropout(float(dropout)),
        )
        self.gate2 = GRUGate(d_model)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        norm_x = self.ln1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, attn_mask=attn_mask)
        x = self.gate1(x, attn_out)
        ff_out = self.ff(self.ln2(x))
        return self.gate2(x, ff_out)


class BLBStage2SequentialPolicy(nn.Module):
    """v2-scale GTrXL actor + critic over per-step decisions.

    Forward signature:
        ``state``: ``[B, state_dim]``
        Returns ``logits[B, max_step_dim, max_num_levels]`` and ``value[B]``.

    Sampling and log-prob evaluation accept an additional ``slot_mask`` and
    ``per_slot_num_levels`` so the same head can serve every step type.
    """

    def __init__(self, cfg: SequentialPolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_step = nn.Embedding(cfg.horizon, cfg.step_embed_dim)
        self.embed_layer = nn.Embedding(cfg.num_layers, cfg.layer_embed_dim)
        self.embed_block = nn.Embedding(cfg.block_count, cfg.block_embed_dim)
        # One table with slot-specific index offsets is equivalent to one
        # embedding per slot, but avoids launching many tiny embedding kernels
        # during the 59-step rollout loop.
        self.prev_action_embedding = nn.Embedding(
            cfg.max_step_dim * cfg.max_num_levels,
            cfg.prev_action_embed_dim,
        )
        self.register_buffer(
            "_prev_action_slot_offsets",
            torch.arange(cfg.max_step_dim, dtype=torch.long) * int(cfg.max_num_levels),
            persistent=False,
        )
        self.register_buffer(
            "_level_indices",
            torch.arange(cfg.max_num_levels, dtype=torch.long).view(1, 1, -1),
            persistent=False,
        )
        action_decode_scales = torch.full(
            (cfg.max_step_dim,), 8.0, dtype=torch.float32,
        )
        if cfg.metadata_width == 0 and cfg.signal_width == 4:
            # Canonical layerwise observations encode fusion by /1 and K by /5.
            action_decode_scales.fill_(float(max(1, cfg.max_num_levels - 1)))
            action_decode_scales[0] = 1.0
        self.register_buffer(
            "_action_history_decode_scales",
            action_decode_scales,
            persistent=False,
        )
        steps, layers, blocks = self._make_step_layer_block_indices(cfg)
        self.register_buffer("_step_indices", steps, persistent=False)
        self.register_buffer("_layer_indices", layers, persistent=False)
        self.register_buffer("_block_indices", blocks, persistent=False)
        self.fc_continuous = nn.Sequential(
            nn.Linear(4 + cfg.signal_width + 1, cfg.cont_proj_dim),
            nn.LayerNorm(cfg.cont_proj_dim),
            nn.SiLU(),
        )
        token_input_dim = (
            cfg.step_embed_dim
            + cfg.layer_embed_dim
            + cfg.block_embed_dim
            + cfg.max_step_dim * cfg.prev_action_embed_dim
            + cfg.cont_proj_dim
        )
        self.input_proj = (
            nn.Identity()
            if token_input_dim == cfg.d_model
            else nn.Linear(token_input_dim, cfg.d_model)
        )
        self.gtrxl_blocks = nn.ModuleList([
            SequentialGTrXLBlock(
                cfg.d_model,
                cfg.n_heads,
                cfg.d_ff,
                cfg.dropout,
            )
            for _ in range(cfg.n_layers)
        ])
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.actor_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.actor_dim),
            nn.Tanh(),
        )
        # Per-slot actor heads, vectorized as one parameter tensor. This keeps
        # the architecture's independent slot heads while replacing many small
        # Linear calls with one batched contraction.
        self.slot_head_weight = nn.Parameter(torch.empty(
            cfg.max_step_dim,
            cfg.max_num_levels,
            cfg.actor_dim,
        ))
        self.slot_head_bias = nn.Parameter(torch.zeros(
            cfg.max_step_dim,
            cfg.max_num_levels,
        ))
        self.value_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.critic_dim),
            nn.Tanh(),
            nn.Linear(cfg.critic_dim, 1),
        )
        self.return_normalizer = RunningMeanStd()
        self._ppo_lr_scale = 1.0
        self._ppo_last_avg_kl = -1.0
        self.default_prior_scale = float(cfg.default_prior_scale)
        self.register_buffer(
            "_preferred_per_slot_idx",
            torch.full((cfg.max_step_dim,), -1, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_preferred_prior_template",
            torch.zeros(cfg.max_step_dim, cfg.max_num_levels, dtype=torch.float32),
            persistent=False,
        )
        # ADR-012 per-slot exploration floor (fusion mode). Mixture sampling
        # pi' = (1-eps)*pi + eps*Uniform(masked support) is the TRUE policy
        # for those slots — sample_action AND evaluate_action (PPO replay) use
        # the same mixture, so log-prob ratios stay consistent. Guarantees the
        # policy can never become deterministic on the fusion option choice
        # (the 2nd 60k ended at entropy 0.000 / clip 0.000 — frozen for the
        # last 18k episodes). Zeros (default) = exact pre-ADR-012 behavior.
        self.register_buffer(
            "_slot_exploration_epsilon",
            torch.zeros(cfg.max_step_dim, dtype=torch.float32),
            persistent=False,
        )
        self._slot_exploration_enabled = False
        self._causal_mask_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}
        self._init_weights()

    @property
    def slot_heads(self) -> Tuple[_SlotHeadView, ...]:
        return tuple(
            _SlotHeadView(self.slot_head_weight[idx], self.slot_head_bias[idx])
            for idx in range(int(self.cfg.max_step_dim))
        )

    def _init_weights(self) -> None:
        def init_fast_linear(layer: nn.Linear, gain: float = 1.0) -> None:
            nn.init.xavier_uniform_(layer.weight, gain=float(gain))
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

        for module in [self.fc_continuous, self.actor_head, self.value_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    init_fast_linear(layer, gain=float(np.sqrt(2)))
        if isinstance(self.input_proj, nn.Linear):
            init_fast_linear(self.input_proj, gain=1.0)
        for block in self.gtrxl_blocks:
            nn.init.xavier_uniform_(block.attn.in_proj_weight, gain=1.0)
            if block.attn.in_proj_bias is not None:
                nn.init.constant_(block.attn.in_proj_bias, 0.0)
            init_fast_linear(block.attn.out_proj, gain=1.0)
            for layer in list(block.ff) + [
                    block.gate1.linear_r, block.gate1.linear_z, block.gate1.linear_h,
                    block.gate2.linear_r, block.gate2.linear_z, block.gate2.linear_h,
            ]:
                if isinstance(layer, nn.Linear):
                    init_fast_linear(layer, gain=float(np.sqrt(2)))
        nn.init.normal_(self.prev_action_embedding.weight, mean=0.0, std=0.02)
        # v2 trick: keep actor logits near zero, so the external warmstart prior
        # is the dominant early signal even with a large GTrXL trunk.
        for slot_idx in range(int(self.cfg.max_step_dim)):
            nn.init.orthogonal_(self.slot_head_weight[slot_idx], gain=0.01)
        nn.init.constant_(self.slot_head_bias, 0.0)

    # ------------------------------------------------------------------
    @staticmethod
    def _make_step_layer_block_indices(
            cfg: SequentialPolicyConfig,
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        steps = torch.arange(cfg.horizon, dtype=torch.long)
        if cfg.step_layer_indices is not None:
            return (
                steps,
                torch.as_tensor(cfg.step_layer_indices, dtype=torch.long),
                torch.as_tensor(cfg.step_block_indices, dtype=torch.long),
            )
        layers = torch.where(
            steps < 4,
            torch.zeros_like(steps),
            1 + torch.div(steps - 4, 5, rounding_mode="floor"),
        )
        blocks = torch.where(steps < 4, steps + 1, torch.remainder(steps - 4, 5))
        layers = torch.clamp(layers, min=0, max=max(0, cfg.num_layers - 1))
        blocks = torch.clamp(blocks, min=0, max=max(0, cfg.block_count - 1))
        return steps, layers, blocks

    def _step_layer_block_indices(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._step_indices, self._layer_indices, self._block_indices

    def _parse_state(
            self,
            state: torch.Tensor,
            *,
            truncate_to_current: bool = False,
            truncate_seq_len: Optional[int] = None,
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = int(state.shape[0])
        H = int(self.cfg.horizon)
        S = int(self.cfg.max_step_dim)
        device = state.device
        current_step = torch.zeros(B, dtype=torch.long, device=device)
        static = torch.zeros(B, 4, dtype=state.dtype, device=device)
        width = int(state.shape[-1])
        if width >= 4:
            static[:, : min(4, width)] = state[:, : min(4, width)]
        cursor = 4
        if width >= cursor + H:
            step_oh = state[:, cursor: cursor + H]
            current_step = torch.argmax(step_oh, dim=-1).long()
        cursor += H
        cursor += int(self.cfg.metadata_width)
        seq_len = H
        if bool(truncate_to_current) and B == 1:
            if truncate_seq_len is not None:
                seq_len = max(1, min(H, int(truncate_seq_len)))
            else:
                seq_len = int(current_step.detach().clamp(0, H - 1).item()) + 1
        prev_actions = torch.zeros(B, seq_len, S, dtype=torch.long, device=device)
        signal_width = int(self.cfg.signal_width)
        prev_signals = torch.zeros(
            B, seq_len, signal_width, dtype=state.dtype, device=device,
        )
        need_actions = H * S
        if width >= cursor + need_actions:
            raw_actions = state[:, cursor: cursor + seq_len * S].view(B, seq_len, S)
            decode_scales = self._action_history_decode_scales.to(
                device=device, dtype=raw_actions.dtype,
            ).view(1, 1, S)
            prev_actions = torch.round(raw_actions * decode_scales).long().clamp(
                min=0,
                max=max(0, self.cfg.max_num_levels - 1),
            )
        cursor += need_actions
        need_signals = H * signal_width
        if width >= cursor + need_signals:
            prev_signals = state[
                :, cursor: cursor + seq_len * signal_width
            ].view(B, seq_len, signal_width)
        return static, current_step, prev_actions, prev_signals

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (int(seq_len), device)
        cached = self._causal_mask_cache.get(key)
        if cached is not None:
            return cached
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        self._causal_mask_cache[key] = mask
        return mask

    def _build_tokens(
            self,
            state: torch.Tensor,
            *,
            truncate_to_current: bool = False,
            truncate_seq_len: Optional[int] = None,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        static, current_step, prev_actions, prev_signals = self._parse_state(
            state,
            truncate_to_current=bool(truncate_to_current),
            truncate_seq_len=truncate_seq_len,
        )
        B = int(state.shape[0])
        device = state.device
        seq_len = int(prev_actions.shape[1])
        steps, layers, blocks = self._step_layer_block_indices()
        steps = steps[:seq_len]
        layers = layers[:seq_len]
        blocks = blocks[:seq_len]
        step_emb = self.embed_step(steps).unsqueeze(0).expand(B, -1, -1)
        layer_emb = self.embed_layer(layers).unsqueeze(0).expand(B, -1, -1)
        block_emb = self.embed_block(blocks).unsqueeze(0).expand(B, -1, -1)
        slot_offsets = self._prev_action_slot_offsets[: self.cfg.max_step_dim]
        prev_action_indices = prev_actions + slot_offsets.view(1, 1, -1)
        prev_emb = self.prev_action_embedding(prev_action_indices).reshape(
            B,
            seq_len,
            self.cfg.max_step_dim * self.cfg.prev_action_embed_dim,
        )
        static_rep = static.unsqueeze(1).expand(-1, seq_len, -1)
        is_current = F.one_hot(current_step.clamp(0, seq_len - 1), num_classes=seq_len).to(
            dtype=state.dtype,
            device=device,
        ).unsqueeze(-1)
        cont = torch.cat([static_rep, prev_signals, is_current], dim=-1)
        cont_proj = self.fc_continuous(cont)
        token_input = torch.cat([step_emb, layer_emb, block_emb, prev_emb, cont_proj], dim=-1)
        return self.input_proj(token_input), current_step

    def _coerce_prior_scale(
            self,
            baseline_prior_scale: Optional[Any],
            *,
            batch_size: int,
            device: torch.device,
            dtype: torch.dtype,
            ) -> torch.Tensor:
        if baseline_prior_scale is None:
            baseline_prior_scale = float(self.default_prior_scale)
        if isinstance(baseline_prior_scale, torch.Tensor):
            scale = baseline_prior_scale.to(device=device, dtype=dtype).reshape(-1)
        else:
            scale = torch.full(
                (batch_size,),
                float(baseline_prior_scale),
                device=device,
                dtype=dtype,
            )
        if scale.numel() == 1 and batch_size != 1:
            scale = scale.expand(batch_size)
        if scale.numel() != batch_size:
            raise ValueError(
                f"baseline_prior_scale has {scale.numel()} values for batch {batch_size}"
            )
        return scale.view(batch_size, 1, 1)

    def _refresh_preferred_prior_template(self) -> None:
        template = torch.zeros_like(self._preferred_prior_template)
        preferred = self._preferred_per_slot_idx
        valid = (preferred >= 0) & (preferred < int(self.cfg.max_num_levels))
        if bool(valid.any()):
            slot_idx = torch.arange(
                int(self.cfg.max_step_dim),
                device=preferred.device,
                dtype=torch.long,
            )
            template[slot_idx[valid], preferred[valid]] = 1.0
        self._preferred_prior_template.copy_(template)

    def _baseline_prior_logits(
            self,
            *,
            batch_size: int,
            device: torch.device,
            dtype: torch.dtype,
            baseline_prior_scale: Optional[Any],
            ) -> torch.Tensor:
        scale = self._coerce_prior_scale(
            baseline_prior_scale,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        template = self._preferred_prior_template.to(device=device, dtype=dtype)
        return template.unsqueeze(0) * scale

    def forward(
            self,
            state: torch.Tensor,
            baseline_prior_scale: Optional[Any] = None,
            *,
            truncate_to_current: bool = False,
            truncate_seq_len: Optional[int] = None,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        tokens, current_step = self._build_tokens(
            state,
            truncate_to_current=bool(truncate_to_current),
            truncate_seq_len=truncate_seq_len,
        )
        causal_mask = self._get_causal_mask(tokens.size(1), tokens.device)
        x = tokens
        for block in self.gtrxl_blocks:
            x = block(x, attn_mask=causal_mask)
        x = self.ln_final(x)
        batch_idx = torch.arange(x.shape[0], device=x.device)
        h = x[batch_idx, current_step.clamp(0, x.size(1) - 1)]
        actor_feat = self.actor_head(h)
        logits = torch.einsum("ba,sla->bsl", actor_feat, self.slot_head_weight)
        logits = logits + self.slot_head_bias.unsqueeze(0)
        logits = logits + self._baseline_prior_logits(
            batch_size=int(logits.shape[0]),
            device=logits.device,
            dtype=logits.dtype,
            baseline_prior_scale=baseline_prior_scale,
        )
        value = self.value_head(h).squeeze(-1)
        return logits, value

    # ------------------------------------------------------------------
    @staticmethod
    def _build_logit_mask(
            slot_mask: torch.Tensor,
            per_slot_num_levels: torch.Tensor,
            max_num_levels: int,
            action_level_mask: Optional[torch.Tensor] = None,
            level_indices: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
        """Return additive -inf mask for invalid (slot, level) cells.

        ``slot_mask`` is ``[B, max_step_dim]`` boolean (True = active).
        ``per_slot_num_levels`` is ``[B, max_step_dim]`` int (0 for padding).
        ``action_level_mask`` is optional ``[B, max_step_dim, max_num_levels]``
        or ``[max_step_dim, max_num_levels]`` boolean (True = allowed).
        Output has shape ``[B, max_step_dim, max_num_levels]``.
        """
        B, S = slot_mask.shape
        slot_mask_bool = slot_mask.to(dtype=torch.bool)
        if level_indices is None:
            level_indices = torch.arange(
                max_num_levels,
                device=slot_mask.device,
                dtype=torch.long,
            ).view(1, 1, -1)
        else:
            level_indices = level_indices.to(device=slot_mask.device, dtype=torch.long)
        levels_idx = level_indices[:, :, :max_num_levels].expand(B, S, -1)
        # padding-slot rows are entirely -inf
        slot_alive = slot_mask_bool.unsqueeze(-1).expand(-1, -1, max_num_levels)
        # within an active slot, levels >= num_levels[slot] get -inf
        level_valid = levels_idx < per_slot_num_levels.unsqueeze(-1)
        valid = slot_alive & level_valid
        if action_level_mask is not None:
            extra_mask = action_level_mask.to(device=slot_mask.device, dtype=torch.bool)
            if extra_mask.dim() == 2:
                extra_mask = extra_mask.unsqueeze(0)
            if extra_mask.dim() != 3:
                raise ValueError(
                    "action_level_mask must be [B, max_step_dim, max_num_levels] "
                    "or [max_step_dim, max_num_levels]"
                )
            try:
                extra_mask = extra_mask.expand(B, S, max_num_levels)
            except RuntimeError as exc:
                raise ValueError(
                    f"action_level_mask shape {tuple(extra_mask.shape)} is not "
                    f"broadcastable to {(B, S, max_num_levels)}"
                ) from exc
            valid = valid & extra_mask
            if (slot_mask_bool & ~valid.any(dim=-1)).any():
                raise ValueError("action_level_mask leaves an active slot with no allowed levels")
        mask = torch.zeros_like(valid, dtype=torch.float32)
        mask = mask.masked_fill(~valid, float("-inf"))
        return mask

    # ------------------------------------------------------------------
    def sample_action(
            self,
            state: torch.Tensor,
            slot_mask: torch.Tensor,
            per_slot_num_levels: torch.Tensor,
            *,
            deterministic: bool = False,
            generator: Optional[torch.Generator] = None,
            action_level_mask: Optional[torch.Tensor] = None,
            logit_mask: Optional[torch.Tensor] = None,
            baseline_prior_scale: Optional[Any] = None,
            truncate_to_current: bool = False,
            truncate_seq_len: Optional[int] = None,
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample one per-step action.

        Returns:
            actions:  ``[B, max_step_dim]`` long (padding entries are 0)
            log_prob: ``[B]`` summed across active slots
            value:    ``[B]``
        """
        logits, value = self.forward(
            state,
            baseline_prior_scale=baseline_prior_scale,
            truncate_to_current=bool(truncate_to_current),
            truncate_seq_len=truncate_seq_len,
        )
        if logit_mask is None:
            logit_mask = self._build_logit_mask(
                slot_mask, per_slot_num_levels, self.cfg.max_num_levels,
                action_level_mask=action_level_mask,
                level_indices=self._level_indices,
            )
        logits = logits + logit_mask
        # collapse padding rows by setting them to a single dummy distribution
        # so torch.distributions doesn't NaN. We then mask-out their log_prob
        # contribution at the end.
        safe_logits = torch.where(
            torch.isfinite(logits).any(dim=-1, keepdim=True),
            logits,
            torch.zeros_like(logits),
        )
        dist = self._action_dist(logits, safe_logits)
        if deterministic:
            actions = torch.argmax(safe_logits, dim=-1)
        elif generator is not None:
            probs = dist.probs
            flat_probs = probs.reshape(-1, probs.shape[-1])
            actions = torch.multinomial(
                flat_probs,
                num_samples=1,
                replacement=True,
                generator=generator,
            ).reshape(probs.shape[:-1])
        else:
            actions = dist.sample()
        actions = torch.where(
            slot_mask.to(device=actions.device, dtype=torch.bool),
            actions,
            torch.zeros_like(actions),
        )
        log_prob_per_slot = dist.log_prob(actions)            # [B, max_step_dim]
        # zero out log_prob for padding rows
        log_prob_per_slot = log_prob_per_slot * slot_mask.float()
        log_prob = log_prob_per_slot.sum(dim=-1)
        return actions, log_prob, value

    def evaluate_action(
            self,
            state: torch.Tensor,
            actions: torch.Tensor,
            slot_mask: torch.Tensor,
            per_slot_num_levels: torch.Tensor,
            action_level_mask: Optional[torch.Tensor] = None,
            logit_mask: Optional[torch.Tensor] = None,
            baseline_prior_scale: Optional[Any] = None,
            return_per_slot_entropy: bool = False,
            return_per_slot_log_prob: bool = False,
            truncate_to_current: bool = False,
            truncate_seq_len: Optional[int] = None,
            ) -> Tuple[torch.Tensor, ...]:
        """Re-evaluate (log_prob, entropy, value) for a given action under the
        current policy. Used by PPO update.
        """
        logits, value = self.forward(
            state,
            baseline_prior_scale=baseline_prior_scale,
            truncate_to_current=bool(truncate_to_current),
            truncate_seq_len=truncate_seq_len,
        )
        if logit_mask is None:
            logit_mask = self._build_logit_mask(
                slot_mask, per_slot_num_levels, self.cfg.max_num_levels,
                action_level_mask=action_level_mask,
                level_indices=self._level_indices,
            )
        logits = logits + logit_mask
        safe_logits = torch.where(
            torch.isfinite(logits).any(dim=-1, keepdim=True),
            logits,
            torch.zeros_like(logits),
        )
        dist = self._action_dist(logits, safe_logits)
        actions_long = actions.long()
        log_prob_per_slot = dist.log_prob(actions_long) * slot_mask.float()
        log_prob = log_prob_per_slot.sum(dim=-1)
        entropy_per_slot = dist.entropy() * slot_mask.float()
        entropy = entropy_per_slot.sum(dim=-1)
        if return_per_slot_entropy and return_per_slot_log_prob:
            return log_prob, entropy, value, entropy_per_slot, log_prob_per_slot
        if return_per_slot_entropy:
            return log_prob, entropy, value, entropy_per_slot
        if return_per_slot_log_prob:
            return log_prob, entropy, value, log_prob_per_slot
        return log_prob, entropy, value

    def set_slot_exploration_epsilon(self, eps_per_slot) -> None:
        """Install the ADR-012 per-slot exploration floor (0 = off)."""
        eps = torch.as_tensor(list(eps_per_slot), dtype=torch.float32)
        if eps.numel() != self.cfg.max_step_dim:
            raise ValueError(
                f"eps length {eps.numel()} != max_step_dim {self.cfg.max_step_dim}"
            )
        if bool((eps < 0).any()) or bool((eps >= 1).any()):
            raise ValueError("exploration epsilon must be in [0, 1)")
        self._slot_exploration_enabled = bool((eps > 0).any().item())
        with torch.no_grad():
            self._slot_exploration_epsilon.copy_(eps.to(self._slot_exploration_epsilon.device))

    def _action_dist(
            self,
            logits: torch.Tensor,
            safe_logits: torch.Tensor,
            ) -> torch.distributions.Categorical:
        """Categorical over masked logits, with the ADR-012 epsilon mixture.

        ``logits`` still carries -inf on masked levels (defines the uniform
        support); ``safe_logits`` is the NaN-safe version used for softmax.
        With all epsilons 0 this reduces exactly to Categorical(safe_logits).
        """
        if not self._slot_exploration_enabled:
            return torch.distributions.Categorical(logits=safe_logits)
        eps = self._slot_exploration_epsilon
        probs = torch.softmax(safe_logits, dim=-1)
        allowed = torch.isfinite(logits).float()
        denom = allowed.sum(dim=-1, keepdim=True)
        # padding rows (no allowed level) fall back to all-uniform; their
        # log-prob contribution is masked out by slot_mask downstream.
        uniform = torch.where(
            denom > 0, allowed / denom.clamp(min=1.0),
            torch.full_like(allowed, 1.0 / float(allowed.shape[-1])),
        )
        e = eps.view(1, -1, 1).to(probs.dtype)
        return torch.distributions.Categorical(probs=(1.0 - e) * probs + e * uniform)

    def apply_preferred_per_step_bias(
            self,
            preferred_per_slot_idx: Sequence[int],
            *,
            gain: float = 1.5,
            clear_existing: bool = True,
            ) -> None:
        """Install an external per-slot warmstart prior.

        Older MLP policy builds wrote this bias directly into the actor head.
        The GTrXL policy keeps learned logits near zero and adds a caller-owned
        prior in ``forward``. That makes the prior schedulable per episode and
        replayable during PPO updates via ``baseline_prior_scale``.
        """
        if len(preferred_per_slot_idx) != self.cfg.max_step_dim:
            raise ValueError(
                f"preferred length {len(preferred_per_slot_idx)} != max_step_dim {self.cfg.max_step_dim}"
            )
        with torch.no_grad():
            preferred = torch.full_like(self._preferred_per_slot_idx, -1)
            for slot_idx, lvl in enumerate(preferred_per_slot_idx):
                lvl = int(lvl)
                if lvl == -1:
                    # -1 == no prior on this slot (the internal sentinel).
                    # ADR-011: fusion mode uses this for the option slot so the
                    # decayed-but-permanent baseline prior stops pulling the
                    # 2-way fusion choice back to option 0 after the anchor.
                    continue
                if lvl < 0 or lvl >= self.cfg.max_num_levels:
                    raise ValueError(
                        f"preferred index {lvl} out of range [0, {self.cfg.max_num_levels})"
                    )
                preferred[int(slot_idx)] = int(lvl)
            self._preferred_per_slot_idx.copy_(preferred)
            self._refresh_preferred_prior_template()
            self.default_prior_scale = float(gain)

    def set_initial_slot_probabilities(
            self,
            probabilities_by_slot: Sequence[Mapping[int, float]],
            action_values_by_index: Sequence[Sequence[int]],
            ) -> None:
        """Initialize independent slot heads from decoded-value probabilities.

        The decoded action values define each slot's legal support and ordering.
        Head weights are zeroed so these probabilities are exact initially;
        normal gradient updates can subsequently learn both weights and biases.
        Any preferred-action warmstart prior is cleared to avoid adding a hidden
        logit offset to this explicit full-support initialization.
        """
        slot_count = int(self.cfg.max_step_dim)
        if len(probabilities_by_slot) != slot_count:
            raise ValueError(
                f"probabilities_by_slot has {len(probabilities_by_slot)} slots, "
                f"expected {slot_count}"
            )
        if len(action_values_by_index) != slot_count:
            raise ValueError(
                f"action_values_by_index has {len(action_values_by_index)} slots, "
                f"expected {slot_count}"
            )

        log_probabilities: List[Tuple[float, ...]] = []
        for slot_idx, (probability_map, action_values) in enumerate(zip(
                probabilities_by_slot, action_values_by_index,
        )):
            if not isinstance(probability_map, Mapping):
                raise ValueError(f"slot {slot_idx} probabilities must be a mapping")
            values = tuple(int(value) for value in action_values)
            if not values or len(values) > int(self.cfg.max_num_levels):
                raise ValueError(
                    f"slot {slot_idx} support size {len(values)} must be in "
                    f"[1, {self.cfg.max_num_levels}]"
                )
            if len(set(values)) != len(values):
                raise ValueError(f"slot {slot_idx} action values must be unique")
            if set(probability_map.keys()) != set(values):
                raise ValueError(
                    f"slot {slot_idx} probability keys must exactly match "
                    f"decoded support {values}"
                )
            probabilities = tuple(float(probability_map[value]) for value in values)
            if any(not math.isfinite(probability) or probability <= 0.0
                   for probability in probabilities):
                raise ValueError(
                    f"slot {slot_idx} probabilities must be finite and strictly positive"
                )
            if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-8):
                raise ValueError(
                    f"slot {slot_idx} probabilities must sum to 1, "
                    f"got {sum(probabilities):.12g}"
                )
            log_probabilities.append(tuple(math.log(value) for value in probabilities))

        with torch.no_grad():
            self.slot_head_weight.zero_()
            self.slot_head_bias.zero_()
            for slot_idx, values in enumerate(log_probabilities):
                self.slot_head_bias[slot_idx, :len(values)] = torch.as_tensor(
                    values,
                    dtype=self.slot_head_bias.dtype,
                    device=self.slot_head_bias.device,
                )
            self._preferred_per_slot_idx.fill_(-1)
            self._refresh_preferred_prior_template()
            self.default_prior_scale = 0.0
            self._slot_exploration_epsilon.zero_()
            self._slot_exploration_enabled = False

    def ppo_aux_state_dict(self) -> Dict[str, Any]:
        """State not owned by ``nn.Module.state_dict`` but needed for resume."""
        return {
            "return_normalizer": self.return_normalizer.state_dict(),
            "ppo_lr_scale": float(self._ppo_lr_scale),
            "ppo_last_avg_kl": float(self._ppo_last_avg_kl),
            "default_prior_scale": float(self.default_prior_scale),
            "preferred_per_slot_idx": self._preferred_per_slot_idx.detach().cpu().tolist(),
        }

    def load_ppo_aux_state_dict(self, state: Optional[Mapping[str, Any]]) -> None:
        if not isinstance(state, Mapping):
            return
        self.return_normalizer.load_state_dict(state.get("return_normalizer"))
        self._ppo_lr_scale = float(state.get("ppo_lr_scale", self._ppo_lr_scale))
        self._ppo_last_avg_kl = float(state.get("ppo_last_avg_kl", self._ppo_last_avg_kl))
        self.default_prior_scale = float(
            state.get("default_prior_scale", self.default_prior_scale)
        )
        preferred = state.get("preferred_per_slot_idx")
        if preferred is not None:
            arr = torch.as_tensor(preferred, dtype=torch.long, device=self._preferred_per_slot_idx.device)
            if arr.numel() == self._preferred_per_slot_idx.numel():
                self._preferred_per_slot_idx.copy_(arr.view_as(self._preferred_per_slot_idx))
                self._refresh_preferred_prior_template()


# ---------------------------------------------------------------------------
# Rollout buffer + GAE
# ---------------------------------------------------------------------------

@dataclass
class SequentialTransition:
    state: np.ndarray             # [state_dim]
    action: np.ndarray            # [max_step_dim]
    slot_mask: np.ndarray         # [max_step_dim] bool
    per_slot_num_levels: np.ndarray  # [max_step_dim] int
    log_prob: Any
    value: Any
    reward: float
    done: bool
    # action_level_mask: np.ndarray when provided; shape [max_step_dim, max_num_levels].
    action_level_mask: Optional[np.ndarray] = None
    baseline_prior_scale: float = 0.0
    actor_cost_per_slot: Optional[np.ndarray] = None
    actor_shared_return: Optional[float] = None


def _compute_gae_from_arrays(
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        *,
        gamma: float,
        lam: float,
        ) -> Tuple[np.ndarray, np.ndarray]:
    n = int(rewards.shape[0])
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in range(n - 1, -1, -1):
        if bool(dones[t]) or t == n - 1:
            next_value = 0.0
        else:
            next_value = float(values[t + 1])
        not_done = 0.0 if bool(dones[t]) else 1.0
        delta = float(rewards[t]) + float(gamma) * next_value * not_done - float(values[t])
        last_gae = delta + float(gamma) * float(lam) * not_done * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return returns, advantages


def _pack_transition_scalar_tensors(
        transitions: Sequence[SequentialTransition],
        field_name: str,
        device: torch.device,
        ) -> torch.Tensor:
    values = [getattr(t, field_name) for t in transitions]
    if not any(torch.is_tensor(value) for value in values):
        return torch.as_tensor([float(value) for value in values], dtype=torch.float32, device=device)
    tensor_device = next(
        (value.device for value in values if torch.is_tensor(value)),
        device,
    )
    packed = torch.stack([
        (
            value.detach().reshape(()).to(
                device=tensor_device, dtype=torch.float32, non_blocking=True,
            )
            if torch.is_tensor(value)
            else torch.tensor(float(value), dtype=torch.float32, device=tensor_device)
        )
        for value in values
    ], dim=0)
    return packed.to(device=device, dtype=torch.float32, non_blocking=True)


def _compute_gae_from_tensors(
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: np.ndarray,
        *,
        gamma: float,
        lam: float,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    n = int(rewards.shape[0])
    advantages = torch.zeros_like(rewards, dtype=torch.float32)
    last_gae = torch.zeros((), dtype=torch.float32, device=rewards.device)
    gamma_f = float(gamma)
    lam_f = float(lam)
    for t in range(n - 1, -1, -1):
        done = bool(dones[t])
        not_done = 0.0 if done else 1.0
        next_value = (
            torch.zeros((), dtype=torch.float32, device=rewards.device)
            if done or t == n - 1
            else values[t + 1]
        )
        delta = rewards[t] + gamma_f * next_value * not_done - values[t]
        last_gae = delta + gamma_f * lam_f * not_done * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return returns, advantages


class SequentialRolloutBuffer:
    """Stores transitions across multiple horizon-N episodes; computes GAE.

    Layout: a flat list of ``SequentialTransition``. Episodes are separated
    by ``done=True`` markers (the transition ON which the episode ended).
    """

    def __init__(self):
        self._buf: List[SequentialTransition] = []

    def __len__(self) -> int:
        return len(self._buf)

    def add(
            self,
            *,
            state: np.ndarray,
            action: np.ndarray,
            slot_mask: np.ndarray,
            per_slot_num_levels: np.ndarray,
            action_level_mask: Optional[np.ndarray] = None,
            log_prob: Any,
            value: Any,
            reward: float,
            done: bool,
            baseline_prior_scale: float = 0.0,
            ) -> int:
        """Append one transition.

        Callers that build a per-step NumPy mask can pass
        ``action_level_mask=action_level_mask_np``. ``None`` preserves the
        original unmasked per-level support.
        """
        self._buf.append(SequentialTransition(
            state=np.asarray(state, dtype=np.float32),
            action=np.asarray(action, dtype=np.int64),
            slot_mask=np.asarray(slot_mask, dtype=bool),
            per_slot_num_levels=np.asarray(per_slot_num_levels, dtype=np.int64),
            action_level_mask=(
                None if action_level_mask is None
                else np.asarray(action_level_mask, dtype=bool)
            ),
            log_prob=log_prob.detach().reshape(()) if torch.is_tensor(log_prob) else float(log_prob),
            value=value.detach().reshape(()) if torch.is_tensor(value) else float(value),
            reward=float(reward),
            done=bool(done),
            baseline_prior_scale=float(baseline_prior_scale),
        ))
        return len(self._buf) - 1

    def add_reward_at(self, index: int, delta: float) -> None:
        """Add a delayed reward component to an existing transition."""
        idx = int(index)
        if idx < 0 or idx >= len(self._buf):
            raise IndexError(f"transition index {idx} out of range")
        self._buf[idx].reward = float(self._buf[idx].reward) + float(delta)

    def set_actor_cost_at(self, index: int, per_slot_cost: np.ndarray) -> None:
        """Attach P3 cost credit for each active factor at one transition."""
        idx = int(index)
        if idx < 0 or idx >= len(self._buf):
            raise IndexError(f"transition index {idx} out of range")
        transition = self._buf[idx]
        values = np.asarray(per_slot_cost, dtype=np.float32)
        if values.shape != transition.action.shape:
            raise ValueError(
                f"per-slot actor cost shape {values.shape} != action shape "
                f"{transition.action.shape}"
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("per-slot actor costs must be finite and nonnegative")
        if np.any(values[~transition.slot_mask] != 0.0):
            raise ValueError("inactive action slots cannot receive actor cost credit")
        transition.actor_cost_per_slot = values.copy()

    def set_actor_shared_return_at(self, index: int, shared_return: float) -> None:
        """Attach the terminal constraint return shared by all action factors."""
        idx = int(index)
        if idx < 0 or idx >= len(self._buf):
            raise IndexError(f"transition index {idx} out of range")
        value = float(shared_return)
        if not math.isfinite(value):
            raise ValueError("actor shared return must be finite")
        self._buf[idx].actor_shared_return = value

    def factorized_actor_advantages(
            self,
            scalar_advantages: torch.Tensor,
            device: torch.device,
            ) -> Optional[torch.Tensor]:
        """Build shared-constraint plus own-cost actor credit per factor."""
        if not any(row.actor_cost_per_slot is not None for row in self._buf):
            return None
        if scalar_advantages.ndim != 1 or scalar_advantages.shape[0] != len(self._buf):
            raise ValueError("scalar advantages must contain one value per transition")
        width = int(self._buf[0].action.size)
        costs = np.zeros((len(self._buf), width), dtype=np.float32)
        for index, transition in enumerate(self._buf):
            values = transition.actor_cost_per_slot
            if values is None:
                continue
            if values.shape != (width,):
                raise ValueError("all per-slot actor cost rows must share one width")
            costs[index] = values
        costs_t = torch.from_numpy(costs).to(device=device)
        has_shared_return = [row.actor_shared_return is not None for row in self._buf]
        if any(has_shared_return):
            if not all(has_shared_return):
                raise ValueError("actor shared returns must be set for the whole update window")
            shared = torch.as_tensor(
                [float(row.actor_shared_return) for row in self._buf],
                dtype=scalar_advantages.dtype,
                device=device,
            )
            return shared.unsqueeze(-1) + costs_t
        return (
            scalar_advantages.unsqueeze(-1)
            - costs_t.sum(dim=-1, keepdim=True)
            + costs_t
        )

    def clear(self) -> None:
        self._buf.clear()

    def compute_gae(
            self,
            *,
            gamma: float = 0.99,
            lam: float = 0.95,
            ) -> Tuple[np.ndarray, np.ndarray]:
        """Standard GAE-λ over the buffer.

        For terminal transitions the next-value bootstrap is 0. For
        non-terminal transitions the next-value is the value of the next
        stored transition (we assume the buffer is contiguous within an
        episode).
        """
        n = len(self._buf)
        if n == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
        rewards = np.empty(n, dtype=np.float32)
        values = np.empty(n, dtype=np.float32)
        dones = np.empty(n, dtype=bool)
        for i, t in enumerate(self._buf):
            rewards[i] = float(t.reward)
            values[i] = float(t.value)
            dones[i] = bool(t.done)
        return _compute_gae_from_arrays(
            rewards,
            values,
            dones,
            gamma=float(gamma),
            lam=float(lam),
        )

    def grpo_advantages(
            self,
            *,
            eps: float = 1e-4,
            outlier_clip: float = 0.0,
            ) -> np.ndarray:
        """Per-step group-relative (GRPO) advantage, aligned with ``to_tensors``.

        Outcome-supervised: each episode's total (summed) reward is normalized
        across the whole update window (the GRPO "group" -- every episode starts
        from the same frozen model + static_skeletons baseline) and broadcast to
        every step of that episode. No critic / GAE. See
        ``grpo_common.grpo_per_step_advantages``.
        """
        from grpo_common import grpo_per_step_advantages
        rewards = [t.reward for t in self._buf]
        dones = [t.done for t in self._buf]
        return grpo_per_step_advantages(
            rewards, dones, eps=float(eps), outlier_clip=float(outlier_clip),
        )

    def to_tensors(
            self,
            device: torch.device,
            *,
            gamma: float = 0.99,
            lam: float = 0.95,
            advantage_normalize: bool = True,
            ):
        """Pack everything into batched tensors. Returns:
            states, actions, slot_masks, per_slot_num_levels, level_masks,
            old_log_probs, old_values, returns, advantages, baseline_prior_scales
        """
        if not self._buf:
            raise RuntimeError("SequentialRolloutBuffer is empty")
        (
            states,
            actions,
            slot_masks,
            levels,
            level_masks,
            rewards,
            dones,
            prior_scales,
        ) = self._pack_numpy_arrays()
        log_probs_t = _pack_transition_scalar_tensors(self._buf, "log_prob", device)
        old_values_t = _pack_transition_scalar_tensors(self._buf, "value", device)
        rewards_t = torch.from_numpy(rewards).to(device)
        returns, advantages = _compute_gae_from_tensors(
            rewards_t,
            old_values_t,
            dones,
            gamma=float(gamma),
            lam=float(lam),
        )
        if advantage_normalize and int(advantages.numel()) > 1:
            adv_std_t = advantages.std(unbiased=False)
            normalized_advantages = (advantages - advantages.mean()) / (adv_std_t + 1e-8)
            advantages = torch.where(adv_std_t > 1e-3, normalized_advantages, advantages)
        return (
            torch.from_numpy(states).to(device),
            torch.from_numpy(actions).to(device),
            torch.from_numpy(slot_masks).to(device),
            torch.from_numpy(levels).to(device),
            None if level_masks is None else torch.from_numpy(level_masks).to(device),
            log_probs_t,
            old_values_t,
            returns,
            advantages,
            torch.from_numpy(prior_scales).to(device),
        )

    def _pack_numpy_arrays(
            self,
            ) -> Tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                Optional[np.ndarray],
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ]:
        buf = self._buf
        if not buf:
            raise RuntimeError("SequentialRolloutBuffer is empty")
        n = len(buf)
        first = buf[0]
        states = np.empty((n,) + tuple(first.state.shape), dtype=np.float32)
        actions = np.empty((n,) + tuple(first.action.shape), dtype=np.int64)
        slot_masks = np.empty((n,) + tuple(first.slot_mask.shape), dtype=bool)
        levels = np.empty((n,) + tuple(first.per_slot_num_levels.shape), dtype=np.int64)
        rewards = np.empty(n, dtype=np.float32)
        dones = np.empty(n, dtype=bool)
        prior_scales = np.empty(n, dtype=np.float32)

        level_masks = None

        for i, t in enumerate(buf):
            states[i] = t.state
            actions[i] = t.action
            slot_masks[i] = t.slot_mask
            levels[i] = t.per_slot_num_levels
            rewards[i] = float(t.reward)
            dones[i] = bool(t.done)
            prior_scales[i] = float(t.baseline_prior_scale)
            if t.action_level_mask is not None:
                if level_masks is None:
                    level_masks = np.ones(
                        (n,) + tuple(t.action_level_mask.shape),
                        dtype=bool,
                    )
                level_masks[i] = t.action_level_mask
        return (
            states,
            actions,
            slot_masks,
            levels,
            level_masks,
            rewards,
            dones,
            prior_scales,
        )


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

@dataclass
class SequentialPPOConfig:
    lr: float = 5e-5
    clip_range: float = 0.2
    n_epochs: int = 4
    minibatch_size: int = 2048
    ent_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    value_clip_range: float = 0.2
    normalize_returns: bool = True
    robust_advantage_norm: bool = False
    adv_outlier_clip: float = 6.0
    use_kl_early_stop: bool = True
    kl_target: float = 0.02
    adaptive_lr_kl: bool = False
    kl_adaptive_min_ratio: float = 0.25
    kl_adaptive_max_ratio: float = 1.25
    nonfinite_lr_backoff: float = 0.5
    per_slot_entropy_recovery: bool = False
    per_slot_entropy_floor_frac: float = 0.22
    per_slot_entropy_recovery_multiplier: float = 6.0
    factorized_actor_clip: bool = False
    entropy_average_active_slots: bool = False
    entropy_normalize_active_slots: bool = False


def _factorized_clipped_policy_loss(
        new_log_prob_per_slot: torch.Tensor,
        old_log_prob_per_slot: torch.Tensor,
        advantages: torch.Tensor,
        active_slots: torch.Tensor,
        *,
        clip_range: float,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """PPO-clip each active MultiDiscrete factor independently."""
    if new_log_prob_per_slot.shape != old_log_prob_per_slot.shape:
        raise ValueError("new and old per-slot log probabilities must align")
    if active_slots.shape != new_log_prob_per_slot.shape:
        raise ValueError("active slot mask must align with per-slot log probabilities")
    if advantages.ndim == 1:
        if advantages.shape[0] != new_log_prob_per_slot.shape[0]:
            raise ValueError("advantages must have one scalar per transition")
        slot_advantages = advantages.unsqueeze(-1)
    elif advantages.shape == new_log_prob_per_slot.shape:
        slot_advantages = advantages
    else:
        raise ValueError(
            "advantages must be one scalar per transition or one value per slot"
        )
    active = active_slots.to(dtype=new_log_prob_per_slot.dtype)
    denominator = torch.clamp(active.sum(), min=1.0)
    ratio = torch.exp(new_log_prob_per_slot - old_log_prob_per_slot)
    unclipped = ratio * slot_advantages
    clipped = torch.clamp(
        ratio, 1.0 - float(clip_range), 1.0 + float(clip_range),
    ) * slot_advantages
    loss = -(torch.min(unclipped, clipped) * active).sum() / denominator
    return loss, ratio


def _factorized_approx_kl(
        new_log_prob_per_slot: torch.Tensor,
        old_log_prob_per_slot: torch.Tensor,
        active_slots: torch.Tensor,
        ) -> torch.Tensor:
    """Estimate behavior KL on the same per-factor scale as PPO clipping."""
    if new_log_prob_per_slot.shape != old_log_prob_per_slot.shape:
        raise ValueError("new and old per-slot log probabilities must align")
    if active_slots.shape != new_log_prob_per_slot.shape:
        raise ValueError("active slot mask must align with per-slot log probabilities")
    active = active_slots.to(dtype=new_log_prob_per_slot.dtype)
    denominator = torch.clamp(active.sum(), min=1.0)
    return ((old_log_prob_per_slot - new_log_prob_per_slot) * active).sum() / denominator


def _active_slot_entropy_objective(
        entropy_per_slot: torch.Tensor,
        per_slot_num_levels: torch.Tensor,
        active_slots: torch.Tensor,
        *,
        normalize_by_levels: bool,
        ) -> torch.Tensor:
    """Average entropy over active factors, optionally on a [0, 1] scale."""
    if entropy_per_slot.shape != active_slots.shape:
        raise ValueError("entropy and active slot mask must align")
    if per_slot_num_levels.shape != entropy_per_slot.shape:
        raise ValueError("level counts and entropy must align")
    active = active_slots.to(dtype=entropy_per_slot.dtype)
    values = entropy_per_slot
    if normalize_by_levels:
        max_entropy = torch.log(
            per_slot_num_levels.to(dtype=entropy_per_slot.dtype).clamp_min(2.0)
        )
        values = values / max_entropy
    return (values * active).sum() / torch.clamp(active.sum(), min=1.0)


def _robust_normalize_advantages(
        advantages: torch.Tensor,
        *,
        outlier_clip: float,
        ) -> torch.Tensor:
    if advantages.numel() <= 1:
        return advantages
    adv = advantages.float()
    median = torch.median(adv)
    mad = torch.median(torch.abs(adv - median))
    mad_ok = torch.isfinite(mad) & (mad > 1e-8)
    radius = float(outlier_clip) * 1.4826 * mad
    clipped_adv = torch.clamp(adv, median - radius, median + radius)
    adv = torch.where(mad_ok, clipped_adv, adv)
    std = torch.std(adv, unbiased=False)
    std_ok = torch.isfinite(std) & (std > 1e-8)
    normalized_adv = (adv - torch.mean(adv)) / (std + 1e-8)
    adv = torch.where(std_ok, normalized_adv, adv)
    return adv


def _apply_adaptive_kl_lr(
        policy: BLBStage2SequentialPolicy,
        optimizer: torch.optim.Optimizer,
        cfg: SequentialPPOConfig,
        ) -> Tuple[float, float]:
    if not bool(getattr(cfg, "adaptive_lr_kl", True)):
        policy._ppo_lr_scale = 1.0
    else:
        last_kl = float(getattr(policy, "_ppo_last_avg_kl", -1.0))
        target = max(1e-8, float(getattr(cfg, "kl_target", 0.02)))
        scale = float(getattr(policy, "_ppo_lr_scale", 1.0))
        if last_kl > 1.5 * target:
            scale *= 0.5
        elif 0.0 <= last_kl < 0.5 * target:
            scale *= 1.2
        policy._ppo_lr_scale = float(np.clip(
            scale,
            float(getattr(cfg, "kl_adaptive_min_ratio", 0.25)),
            float(getattr(cfg, "kl_adaptive_max_ratio", 2.5)),
        ))
    lr = float(cfg.lr) * float(policy._ppo_lr_scale)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr, float(policy._ppo_lr_scale)


def sequential_ppo_update(
        policy: BLBStage2SequentialPolicy,
        optimizer: torch.optim.Optimizer,
        buffer: SequentialRolloutBuffer,
        cfg: SequentialPPOConfig,
        device: torch.device,
        ent_coef_override: Optional[float] = None,
        ) -> dict:
    """Run a PPO-clip update over the buffer's transitions.

    ``ent_coef_override``: if not None, replace ``cfg.ent_coef`` for THIS
    update only. Used by the entropy-schedule mechanism in
    :func:`train_sequential` to ramp ent_coef from 0 (anchor) to the target
    value (steady) — see the 2026-05-18 warmstart-sampling bug fix for why
    the schedule matters (PPO entropy bonus was undoing the warmstart bias
    during the forced-baseline anchor episodes, leaving the policy too
    diffuse to land near baseline once sampling started).
    """
    if len(buffer) == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                "clip_fraction": 0.0, "n_samples": 0, "ent_coef": 0.0,
                "approx_kl": 0.0, "kl_early_stop": False,
                "lr": float(cfg.lr), "lr_scale": 1.0,
                "entropy_recovery_delta": 0.0,
                "return_mean": 0.0, "return_std": 1.0,
                "actor_clip_mode": (
                    "factorized_per_slot"
                    if bool(getattr(cfg, "factorized_actor_clip", False)) else "joint"
                ),
                "actor_credit_mode": "scalar_gae",
                "entropy_objective_mode": "joint_sum"}
    effective_ent_coef = (
        float(cfg.ent_coef) if ent_coef_override is None else float(ent_coef_override)
    )

    (
        states,
        actions,
        slot_masks,
        levels,
        level_masks,
        old_log_probs,
        old_values,
        returns,
        advantages,
        prior_scales,
    ) = buffer.to_tensors(
        device, gamma=cfg.gamma, lam=cfg.gae_lambda,
        advantage_normalize=False,
    )
    return_normalizer = getattr(policy, "return_normalizer", None)
    if bool(getattr(cfg, "normalize_returns", True)) and return_normalizer is None:
        return_normalizer = RunningMeanStd()
        setattr(policy, "return_normalizer", return_normalizer)
    normalize_returns = bool(getattr(cfg, "normalize_returns", True)) and return_normalizer is not None
    if normalize_returns:
        return_normalizer.update(returns)
    factorized_actor_clip = bool(getattr(cfg, "factorized_actor_clip", False))
    factorized_actor_advantages = (
        buffer.factorized_actor_advantages(advantages, device)
        if factorized_actor_clip else None
    )
    robust_advantage_norm = bool(getattr(cfg, "robust_advantage_norm", True))
    if factorized_actor_advantages is not None:
        centered_factorized = factorized_actor_advantages.clone()
        for slot_idx in range(centered_factorized.shape[1]):
            slot_active = slot_masks[:, slot_idx]
            if bool(slot_active.any()):
                slot_values = centered_factorized[slot_active, slot_idx]
                centered_factorized[slot_active, slot_idx] = (
                    slot_values - slot_values.mean()
                )
        active_advantages = centered_factorized[slot_masks]
        if robust_advantage_norm:
            active_advantages = _robust_normalize_advantages(
                active_advantages,
                outlier_clip=float(getattr(cfg, "adv_outlier_clip", 6.0)),
            )
        elif active_advantages.numel() > 1:
            std = torch.std(active_advantages, unbiased=False)
            std_ok = torch.isfinite(std) & (std > 1e-8)
            normalized = (
                active_advantages - torch.mean(active_advantages)
            ) / (std + 1e-8)
            active_advantages = torch.where(std_ok, normalized, active_advantages)
        normalized_factorized = torch.zeros_like(factorized_actor_advantages)
        normalized_factorized[slot_masks] = active_advantages
        factorized_actor_advantages = normalized_factorized
    elif robust_advantage_norm:
        advantages = _robust_normalize_advantages(
            advantages,
            outlier_clip=float(getattr(cfg, "adv_outlier_clip", 6.0)),
        )
    elif advantages.numel() > 1:
        std = torch.std(advantages, unbiased=False)
        std_ok = torch.isfinite(std) & (std > 1e-8)
        normalized_advantages = (advantages - torch.mean(advantages)) / (std + 1e-8)
        advantages = torch.where(std_ok, normalized_advantages, advantages)

    old_log_probs_per_slot = None
    if factorized_actor_clip:
        with torch.no_grad():
            old_eval_out = policy.evaluate_action(
                states,
                actions,
                slot_masks,
                levels,
                action_level_mask=level_masks,
                baseline_prior_scale=prior_scales,
                return_per_slot_entropy=True,
                return_per_slot_log_prob=True,
            )
        if len(old_eval_out) != 5:
            raise RuntimeError(
                "factorized PPO requires evaluate_action per-slot log probabilities"
            )
        old_log_probs_per_slot = old_eval_out[4].detach()
        reconstructed_joint = old_log_probs_per_slot.sum(dim=-1)
        if not bool(torch.allclose(
                reconstructed_joint, old_log_probs, rtol=1.0e-5, atol=1.0e-6,
        )):
            raise RuntimeError(
                "factorized behavior log probabilities do not match rollout joint log probabilities"
            )

    n = states.shape[0]
    current_lr, lr_scale = _apply_adaptive_kl_lr(policy, optimizer, cfg)

    metrics_sum_t = {
        "policy_loss": torch.zeros((), device=device),
        "value_loss": torch.zeros((), device=device),
        "entropy": torch.zeros((), device=device),
        "clip_fraction": torch.zeros((), device=device),
        "approx_kl": torch.zeros((), device=device),
        "entropy_recovery_delta": torch.zeros((), device=device),
    }
    n_minibatches = 0
    kl_early_stop = False
    nonfinite_minibatches = 0

    def _backoff_after_nonfinite() -> None:
        nonlocal current_lr, lr_scale
        min_ratio = float(getattr(cfg, "kl_adaptive_min_ratio", 0.25))
        backoff = float(getattr(cfg, "nonfinite_lr_backoff", 0.5))
        lr_scale = float(max(min_ratio, float(lr_scale) * max(0.0, backoff)))
        policy._ppo_lr_scale = lr_scale
        current_lr = float(cfg.lr) * lr_scale
        for group in optimizer.param_groups:
            group["lr"] = current_lr

    for _ in range(int(cfg.n_epochs)):
        epoch_indices = torch.randperm(n, device=device)
        mb_size = max(1, int(cfg.minibatch_size))
        for start in range(0, n, mb_size):
            end = min(n, start + mb_size)
            mb = epoch_indices[start:end]
            mb_states = states.index_select(0, mb)
            mb_actions = actions.index_select(0, mb)
            mb_slot_masks = slot_masks.index_select(0, mb)
            mb_levels = levels.index_select(0, mb)
            mb_level_masks = (
                None if level_masks is None else level_masks.index_select(0, mb)
            )
            mb_prior_scales = prior_scales.index_select(0, mb)
            if factorized_actor_clip:
                eval_out = policy.evaluate_action(
                    mb_states,
                    mb_actions,
                    mb_slot_masks,
                    mb_levels,
                    action_level_mask=mb_level_masks,
                    baseline_prior_scale=mb_prior_scales,
                    return_per_slot_entropy=True,
                    return_per_slot_log_prob=True,
                )
                (
                    new_log_probs,
                    entropy,
                    value,
                    entropy_per_slot,
                    new_log_probs_per_slot,
                ) = eval_out
            else:
                eval_out = policy.evaluate_action(
                    mb_states,
                    mb_actions,
                    mb_slot_masks,
                    mb_levels,
                    action_level_mask=mb_level_masks,
                    baseline_prior_scale=mb_prior_scales,
                    return_per_slot_entropy=True,
                )
                new_log_probs, entropy, value, entropy_per_slot = eval_out
                new_log_probs_per_slot = None
            old_lp = old_log_probs.index_select(0, mb)
            old_value = old_values.index_select(0, mb)
            ret = returns.index_select(0, mb)
            adv = (
                factorized_actor_advantages.index_select(0, mb)
                if factorized_actor_advantages is not None
                else advantages.index_select(0, mb)
            )
            if factorized_actor_clip:
                old_lp_per_slot = old_log_probs_per_slot.index_select(0, mb)
                policy_loss, ratio = _factorized_clipped_policy_loss(
                    new_log_probs_per_slot,
                    old_lp_per_slot,
                    adv,
                    mb_slot_masks,
                    clip_range=cfg.clip_range,
                )
            else:
                old_lp_per_slot = None
                ratio = torch.exp(new_log_probs - old_lp)
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * adv
                policy_loss = -torch.min(unclipped, clipped).mean()
            if normalize_returns:
                value_target = return_normalizer.normalize(ret)
                value_pred = return_normalizer.normalize(value)
                old_value_pred = return_normalizer.normalize(old_value)
            else:
                value_target = ret
                value_pred = value
                old_value_pred = old_value
            clipped_value = old_value_pred + torch.clamp(
                value_pred - old_value_pred,
                -float(getattr(cfg, "value_clip_range", 1.0)),
                float(getattr(cfg, "value_clip_range", 1.0)),
            )
            value_loss_raw = F.huber_loss(
                value_pred,
                value_target,
                delta=1.0,
                reduction="none",
            )
            value_loss_clipped = F.huber_loss(
                clipped_value,
                value_target,
                delta=1.0,
                reduction="none",
            )
            value_loss = torch.max(value_loss_raw, value_loss_clipped).mean()
            entropy_mean = entropy.mean()
            if (
                    factorized_actor_clip
                    and bool(getattr(cfg, "entropy_average_active_slots", False))
            ):
                entropy_objective = _active_slot_entropy_objective(
                    entropy_per_slot,
                    mb_levels,
                    mb_slot_masks,
                    normalize_by_levels=bool(getattr(
                        cfg, "entropy_normalize_active_slots", False,
                    )),
                )
            else:
                entropy_objective = entropy_mean
            entropy_recovery_delta = torch.zeros((), device=device)
            if bool(getattr(cfg, "per_slot_entropy_recovery", True)):
                mb_slot_masks_float = mb_slot_masks.float()
                mb_levels_float = mb_levels.float().clamp_min(1.0)
                entropy_floor = (
                    float(getattr(cfg, "per_slot_entropy_floor_frac", 0.22))
                    * torch.log(mb_levels_float)
                    * mb_slot_masks_float
                )
                entropy_deficit = (
                    torch.relu(entropy_floor - entropy_per_slot) * mb_slot_masks_float
                )
                denom = torch.clamp(mb_slot_masks_float.sum(), min=1.0)
                entropy_recovery_delta = entropy_deficit.sum() / denom
            if factorized_actor_clip:
                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - effective_ent_coef * entropy_objective
                    - float(getattr(cfg, "per_slot_entropy_recovery_multiplier", 6.0))
                    * entropy_recovery_delta
                )
            else:
                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - effective_ent_coef * entropy_mean
                    - float(getattr(cfg, "per_slot_entropy_recovery_multiplier", 6.0))
                    * entropy_recovery_delta
                )
            finite_tensors = (
                new_log_probs,
                entropy,
                value,
                ratio,
                policy_loss,
                value_loss,
                entropy_mean,
                entropy_objective,
                entropy_recovery_delta,
                loss,
            )
            finite_checks = torch.stack([
                torch.isfinite(t).all().reshape(()) for t in finite_tensors
            ])
            if not bool(finite_checks.all().item()):
                nonfinite_minibatches += 1
                kl_early_stop = True
                optimizer.zero_grad(set_to_none=True)
                _backoff_after_nonfinite()
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.max_grad_norm is not None and cfg.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                if not bool(torch.isfinite(grad_norm).item()):
                    nonfinite_minibatches += 1
                    kl_early_stop = True
                    optimizer.zero_grad(set_to_none=True)
                    _backoff_after_nonfinite()
                    break
            optimizer.step()
            with torch.no_grad():
                if factorized_actor_clip:
                    active_clip_mask = mb_slot_masks.float()
                    clip_frac_t = (
                        (torch.abs(ratio - 1.0) > cfg.clip_range).float()
                        * active_clip_mask
                    ).sum() / torch.clamp(active_clip_mask.sum(), min=1.0)
                else:
                    clip_frac_t = ((torch.abs(ratio - 1.0) > cfg.clip_range).float()).mean()
                if factorized_actor_clip:
                    approx_kl_t = _factorized_approx_kl(
                        new_log_probs_per_slot,
                        old_lp_per_slot,
                        mb_slot_masks,
                    )
                else:
                    approx_kl_t = (old_lp - new_log_probs).mean()
            metrics_sum_t["policy_loss"] += policy_loss.detach()
            metrics_sum_t["value_loss"] += value_loss.detach()
            metrics_sum_t["entropy"] += entropy_mean.detach()
            metrics_sum_t["clip_fraction"] += clip_frac_t.detach()
            metrics_sum_t["approx_kl"] += approx_kl_t.detach()
            metrics_sum_t["entropy_recovery_delta"] += entropy_recovery_delta.detach()
            n_minibatches += 1
        n_seen = max(1, int(n_minibatches))
        epoch_avg_kl_t = metrics_sum_t["approx_kl"] / float(n_seen)
        if nonfinite_minibatches > 0:
            break
        if (
                bool(getattr(cfg, "use_kl_early_stop", True))
                and float(epoch_avg_kl_t.item()) > 1.5 * float(getattr(cfg, "kl_target", 0.02))
        ):
            kl_early_stop = True
            break

    n_mb = max(1, n_minibatches)
    avg_kl_t = metrics_sum_t["approx_kl"] / float(n_mb)
    avg_kl = float(avg_kl_t.item())
    policy._ppo_last_avg_kl = float(avg_kl)
    return {
        "policy_loss": float((metrics_sum_t["policy_loss"] / n_mb).item()),
        "value_loss": float((metrics_sum_t["value_loss"] / n_mb).item()),
        "entropy": float((metrics_sum_t["entropy"] / n_mb).item()),
        "clip_fraction": float((metrics_sum_t["clip_fraction"] / n_mb).item()),
        "approx_kl": float(avg_kl),
        "kl_early_stop": bool(kl_early_stop),
        "n_samples": int(n),
        "ent_coef": float(effective_ent_coef),
        "lr": float(current_lr),
        "lr_scale": float(lr_scale),
        "entropy_recovery_delta": float(
            (metrics_sum_t["entropy_recovery_delta"] / n_mb).item()
        ),
        "return_mean": float(return_normalizer.mean if return_normalizer is not None else 0.0),
        "return_std": float(return_normalizer.std if return_normalizer is not None else 1.0),
        "nonfinite_minibatches": int(nonfinite_minibatches),
        "nonfinite_update_skipped": bool(nonfinite_minibatches > 0 and n_minibatches == 0),
        "actor_clip_mode": (
            "factorized_per_slot" if factorized_actor_clip else "joint"
        ),
        "actor_credit_mode": (
            "shared_constraint_plus_own_cost"
            if factorized_actor_advantages is not None else "scalar_gae"
        ),
        "entropy_objective_mode": (
            "normalized_active_slot_mean"
            if factorized_actor_clip
            and bool(getattr(cfg, "entropy_average_active_slots", False))
            and bool(getattr(cfg, "entropy_normalize_active_slots", False))
            else (
                "active_slot_mean"
                if factorized_actor_clip
                and bool(getattr(cfg, "entropy_average_active_slots", False))
                else "joint_sum"
            )
        ),
    }


def sequential_grpo_update(
        policy: BLBStage2SequentialPolicy,
        reference_policy: Optional[BLBStage2SequentialPolicy],
        optimizer: torch.optim.Optimizer,
        buffer: SequentialRolloutBuffer,
        cfg: SequentialPPOConfig,
        device: torch.device,
        ent_coef_override: Optional[float] = None,
        kl_beta: float = 0.04,
        ) -> dict:
    """Run a GRPO-clip update over the buffer's transitions.

    Minimal swap of PPO's advantage estimator (2026-05-31 PPO->GRPO design):

    * **advantage** = group-relative episode return -- the update window IS the
      group (every episode starts from the same frozen model + static_skeletons
      baseline), broadcast to all of an episode's steps. No GAE, no critic.
    * **no value loss** -- the critic head is left untrained (gets no gradient).
    * **+ ``kl_beta`` * KL(policy || reference)** via the unbiased k3 estimator
      ``exp(Δ)-Δ-1`` (Δ = ref_logp - new_logp) against a frozen reference snapshot.

    The clipped surrogate, entropy bonus, per-slot entropy recovery, KL-adaptive
    LR, early stop, and warmstart-prior replay all match
    :func:`sequential_ppo_update` exactly. The returned metrics keep the same
    keys as PPO (``value_loss``/``return_mean``/... reported as 0) plus
    ``kl_ref``/``kl_beta`` so existing logging/diagnostics keep working.
    """
    raise RuntimeError(
        "GRPO has been disabled for this project after the PPO-vs-GRPO "
        "MRPC generalization study. Use sequential_ppo_update instead."
    )
    empty = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
             "clip_fraction": 0.0, "n_samples": 0, "ent_coef": 0.0,
             "approx_kl": 0.0, "kl_early_stop": False,
             "lr": float(cfg.lr), "lr_scale": 1.0,
             "entropy_recovery_delta": 0.0, "return_mean": 0.0, "return_std": 1.0,
             "kl_ref": 0.0, "kl_beta": float(kl_beta),
             "nonfinite_minibatches": 0, "nonfinite_update_skipped": False}
    if len(buffer) == 0:
        return empty
    effective_ent_coef = (
        float(cfg.ent_coef) if ent_coef_override is None else float(ent_coef_override)
    )

    (
        states, actions, slot_masks, levels, level_masks,
        old_log_probs, _old_values, _returns, _gae_adv, prior_scales,
    ) = buffer.to_tensors(
        device, gamma=cfg.gamma, lam=cfg.gae_lambda, advantage_normalize=False,
    )
    # GRPO group-relative advantage (window = group) replaces GAE entirely.
    adv_np = buffer.grpo_advantages(
        outlier_clip=float(getattr(cfg, "adv_outlier_clip", 6.0)),
    )
    advantages = torch.from_numpy(
        np.ascontiguousarray(adv_np, dtype=np.float32)
    ).to(device)

    n = states.shape[0]
    indices = np.arange(n)
    current_lr, lr_scale = _apply_adaptive_kl_lr(policy, optimizer, cfg)
    use_ref = reference_policy is not None and float(kl_beta) > 0.0

    metrics_sum = {"policy_loss": 0.0, "entropy": 0.0, "clip_fraction": 0.0,
                   "approx_kl": 0.0, "kl_ref": 0.0,
                   "entropy_recovery_delta": 0.0, "n_minibatches": 0}
    kl_early_stop = False
    nonfinite_minibatches = 0

    def _backoff_after_nonfinite() -> None:
        nonlocal current_lr, lr_scale
        min_ratio = float(getattr(cfg, "kl_adaptive_min_ratio", 0.25))
        backoff = float(getattr(cfg, "nonfinite_lr_backoff", 0.5))
        lr_scale = float(max(min_ratio, float(lr_scale) * max(0.0, backoff)))
        policy._ppo_lr_scale = lr_scale
        current_lr = float(cfg.lr) * lr_scale
        for group in optimizer.param_groups:
            group["lr"] = current_lr

    for _ in range(int(cfg.n_epochs)):
        np.random.shuffle(indices)
        mb_size = max(1, int(cfg.minibatch_size))
        for start in range(0, n, mb_size):
            end = min(n, start + mb_size)
            mb = torch.from_numpy(indices[start:end]).long().to(device)
            mb_states = states.index_select(0, mb)
            mb_actions = actions.index_select(0, mb)
            mb_slot_masks = slot_masks.index_select(0, mb)
            mb_levels = levels.index_select(0, mb)
            mb_level_masks = (
                None if level_masks is None else level_masks.index_select(0, mb)
            )
            mb_prior = prior_scales.index_select(0, mb)
            new_log_probs, entropy, _value, entropy_per_slot = policy.evaluate_action(
                mb_states, mb_actions, mb_slot_masks, mb_levels,
                action_level_mask=mb_level_masks,
                baseline_prior_scale=mb_prior,
                return_per_slot_entropy=True,
            )
            old_lp = old_log_probs.index_select(0, mb)
            adv = advantages.index_select(0, mb)
            ratio = torch.exp(new_log_probs - old_lp)
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * adv
            policy_loss = -torch.min(unclipped, clipped).mean()

            # reference-policy KL (k3 unbiased estimator), frozen reference.
            if use_ref:
                with torch.no_grad():
                    ref_out = reference_policy.evaluate_action(
                        mb_states, mb_actions, mb_slot_masks, mb_levels,
                        action_level_mask=mb_level_masks,
                        baseline_prior_scale=mb_prior,
                        return_per_slot_entropy=False,
                    )
                ref_log_probs = ref_out[0]
                delta = ref_log_probs - new_log_probs
                kl_ref = (torch.exp(delta) - delta - 1.0).mean()
            else:
                kl_ref = torch.zeros((), device=device)

            entropy_mean = entropy.mean()
            entropy_recovery_delta = torch.zeros((), device=device)
            if bool(getattr(cfg, "per_slot_entropy_recovery", True)):
                msk = mb_slot_masks.float()
                lvl = mb_levels.float().clamp_min(1.0)
                entropy_floor = (
                    float(getattr(cfg, "per_slot_entropy_floor_frac", 0.22))
                    * torch.log(lvl) * msk
                )
                entropy_deficit = torch.relu(entropy_floor - entropy_per_slot) * msk
                denom = torch.clamp(msk.sum(), min=1.0)
                entropy_recovery_delta = entropy_deficit.sum() / denom
            loss = (
                policy_loss
                - effective_ent_coef * entropy_mean
                - float(getattr(cfg, "per_slot_entropy_recovery_multiplier", 6.0))
                * entropy_recovery_delta
                + float(kl_beta) * kl_ref
            )
            finite_tensors = (new_log_probs, entropy, ratio, policy_loss,
                              entropy_mean, entropy_recovery_delta, kl_ref, loss)
            if not all(bool(torch.isfinite(t).all().item()) for t in finite_tensors):
                nonfinite_minibatches += 1
                kl_early_stop = True
                optimizer.zero_grad(set_to_none=True)
                _backoff_after_nonfinite()
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.max_grad_norm is not None and cfg.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                if not bool(torch.isfinite(grad_norm).item()):
                    nonfinite_minibatches += 1
                    kl_early_stop = True
                    optimizer.zero_grad(set_to_none=True)
                    _backoff_after_nonfinite()
                    break
            optimizer.step()
            with torch.no_grad():
                clip_frac = ((torch.abs(ratio - 1.0) > cfg.clip_range).float()).mean().item()
                approx_kl = (old_lp - new_log_probs).mean().item()
            metrics_sum["policy_loss"] += float(policy_loss.item())
            metrics_sum["entropy"] += float(entropy_mean.item())
            metrics_sum["clip_fraction"] += float(clip_frac)
            metrics_sum["approx_kl"] += float(approx_kl)
            metrics_sum["kl_ref"] += float(kl_ref.detach().item())
            metrics_sum["entropy_recovery_delta"] += float(entropy_recovery_delta.detach().item())
            metrics_sum["n_minibatches"] += 1
        n_seen = max(1, int(metrics_sum["n_minibatches"]))
        epoch_avg_kl = metrics_sum["approx_kl"] / float(n_seen)
        if nonfinite_minibatches > 0:
            break
        if (
                bool(getattr(cfg, "use_kl_early_stop", True))
                and float(epoch_avg_kl) > 1.5 * float(getattr(cfg, "kl_target", 0.02))
        ):
            kl_early_stop = True
            break

    n_mb = max(1, metrics_sum["n_minibatches"])
    avg_kl = metrics_sum["approx_kl"] / n_mb
    policy._ppo_last_avg_kl = float(avg_kl)
    return {
        "policy_loss": metrics_sum["policy_loss"] / n_mb,
        "value_loss": 0.0,
        "entropy": metrics_sum["entropy"] / n_mb,
        "clip_fraction": metrics_sum["clip_fraction"] / n_mb,
        "approx_kl": float(avg_kl),
        "kl_ref": metrics_sum["kl_ref"] / n_mb,
        "kl_beta": float(kl_beta),
        "kl_early_stop": bool(kl_early_stop),
        "n_samples": int(n),
        "ent_coef": float(effective_ent_coef),
        "lr": float(current_lr),
        "lr_scale": float(lr_scale),
        "entropy_recovery_delta": metrics_sum["entropy_recovery_delta"] / n_mb,
        "return_mean": 0.0,
        "return_std": 1.0,
        "nonfinite_minibatches": int(nonfinite_minibatches),
        "nonfinite_update_skipped": bool(nonfinite_minibatches > 0 and metrics_sum["n_minibatches"] == 0),
    }


# ---------------------------------------------------------------------------
# Convenience: derive (slot_mask, per_slot_num_levels) for one BlockStepSpec
# ---------------------------------------------------------------------------

def _spec_slot_num_levels(spec) -> list:
    """Per-slot legal level counts for one step, for either spec type.

    Fusion mode: a FusionStepSpec has 2 slots = (fusion option, K). Legacy
    per-slot mode: a BlockStepSpec exposes per-slot ``slot_dims``.
    """
    if hasattr(spec, "fusion_num_options"):
        return [int(spec.fusion_num_options), int(spec.k_num_levels)]
    return [int(d) for d in spec.slot_dims]


def step_to_mask_and_levels(
        spec,        # BlockStepSpec or FusionStepSpec from action_space
        max_step_dim: int,
        max_num_levels: int,
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (slot_mask[max_step_dim] bool, per_slot_num_levels[max_step_dim] int).

    Active slots = this spec's slot levels, padded to max_step_dim with zeros.
    Per-slot num_levels = each slot's level count (must be <= max_num_levels).
    """
    per_slot = _spec_slot_num_levels(spec)
    n = len(per_slot)
    if n > max_step_dim:
        raise ValueError(f"step has {n} slots > max_step_dim={max_step_dim}")
    slot_mask = np.zeros(max_step_dim, dtype=bool)
    explicit_slot_mask = getattr(spec, "slot_mask", None)
    if explicit_slot_mask is None:
        slot_mask[:n] = True
    else:
        explicit_slot_mask = tuple(bool(value) for value in explicit_slot_mask)
        if len(explicit_slot_mask) != n:
            raise ValueError(
                f"step slot_mask has {len(explicit_slot_mask)} values for {n} slots"
            )
        slot_mask[:n] = explicit_slot_mask
    levels = np.zeros(max_step_dim, dtype=np.int64)
    for i, d in enumerate(per_slot):
        if int(d) > max_num_levels:
            raise ValueError(
                f"slot {i} has {d} levels > max_num_levels={max_num_levels}"
            )
        levels[i] = int(d)
    return slot_mask, levels
