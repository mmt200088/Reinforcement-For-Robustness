"""BLB Stage 2 RL Policy / Value 网络（spec §7.2）。

设计：共享 backbone + 跨层共享头，per-layer embedding 注入位置信息。
对每个分量输出独立 ``Categorical`` logits，整体相当于 ``MultiDiscrete``。

参数规模（默认 d_hidden=256）：
  encoder: state_dim → 256
  per_layer_emb: L × 64
  head:    (256+64) → sum(layer_dims) ≈ 94
  first_input_head: 256 → 5
  value:   256 → 1
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _split_logits_per_dim(
        flat_logits: torch.Tensor,
        dims: Sequence[int],
        ) -> List[torch.Tensor]:
    """``[..., sum(dims)] → [tensor_per_dim, ...]``，每段对应一个 categorical。"""
    sizes = [int(d) for d in dims]
    return list(torch.split(flat_logits, sizes, dim=-1))


def _mask_logits_for_slot(
        logits: torch.Tensor,
        action_mask: Optional[Sequence[Sequence[bool]]],
        slot_idx: int,
        ) -> torch.Tensor:
    if action_mask is None:
        return logits
    if slot_idx >= len(action_mask):
        raise ValueError(f"action_mask has no slot {slot_idx}")
    raw = action_mask[slot_idx]
    mask = torch.as_tensor(raw, dtype=torch.bool, device=logits.device).reshape(-1)
    if mask.numel() != logits.shape[-1]:
        raise ValueError(
            f"action_mask slot {slot_idx} width {mask.numel()} != logits width {logits.shape[-1]}"
        )
    if not bool(mask.any().item()):
        raise ValueError(f"action_mask slot {slot_idx} allows no actions")
    view_shape = [1] * logits.dim()
    view_shape[-1] = int(mask.numel())
    mask = mask.reshape(view_shape)
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


def _adjust_logits_for_slot(
        logits: torch.Tensor,
        action_mask: Optional[Sequence[Sequence[bool]]],
        action_bias: Optional[Sequence[Sequence[float]]],
        slot_idx: int,
        ) -> torch.Tensor:
    out = _mask_logits_for_slot(logits, action_mask, slot_idx)
    if action_bias is None:
        return out
    if slot_idx >= len(action_bias):
        raise ValueError(f"action_bias has no slot {slot_idx}")
    raw = action_bias[slot_idx]
    bias = torch.as_tensor(raw, dtype=out.dtype, device=out.device).reshape(-1)
    if bias.numel() != out.shape[-1]:
        raise ValueError(
            f"action_bias slot {slot_idx} width {bias.numel()} != logits width {out.shape[-1]}"
        )
    view_shape = [1] * out.dim()
    view_shape[-1] = int(bias.numel())
    return out + bias.reshape(view_shape)


@dataclass
class PolicyForwardResult:
    """``BLBStage2Policy.forward`` 的返回。"""
    layer_logits_flat: torch.Tensor       # [..., L * sum(per_layer_dims)]  per-layer 头
    first_input_logits: torch.Tensor      # [..., first_input_levels]
    value: torch.Tensor                   # [...] 标量 critic


class BLBStage2Policy(nn.Module):
    """共享 backbone + per-layer embedding head 的 PPO actor + critic。

    Layout:
      state ──► encoder ──► h ─┐
                                 ├──► per-layer head（layer_emb 拼接，跨层共享参数）
                                 ├──► first_input_head （5 logits）
                                 └──► value head （1）
    每层产出 ``sum(per_layer_dims)`` 个 logits，总动作 logits = L × sum(per_layer_dims) + 5。
    """

    def __init__(
            self,
            *,
            state_dim: int,
            num_layers: int,
            per_layer_dims: Sequence[int],
            first_input_levels: int = 5,
            d_hidden: int = 256,
            d_layer_emb: int = 64,
            ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.num_layers = int(num_layers)
        self.per_layer_dims: List[int] = [int(d) for d in per_layer_dims]
        self.first_input_levels = int(first_input_levels)
        self.d_hidden = int(d_hidden)
        self.d_layer_emb = int(d_layer_emb)

        self.encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.d_hidden),
            nn.ReLU(),
            nn.Linear(self.d_hidden, self.d_hidden),
            nn.ReLU(),
        )
        self.layer_emb = nn.Embedding(self.num_layers, self.d_layer_emb)
        self.layer_head = nn.Sequential(
            nn.Linear(self.d_hidden + self.d_layer_emb, self.d_hidden),
            nn.ReLU(),
            nn.Linear(self.d_hidden, sum(self.per_layer_dims)),
        )
        self.first_input_head = nn.Linear(self.d_hidden, self.first_input_levels)
        self.value_head = nn.Linear(self.d_hidden, 1)

        self._layer_idx_buf: torch.Tensor

    @staticmethod
    def _last_linear(module: nn.Module) -> nn.Linear:
        if isinstance(module, nn.Linear):
            return module
        if isinstance(module, nn.Sequential):
            for child in reversed(module):
                if isinstance(child, nn.Linear):
                    return child
        raise TypeError(f"cannot find final Linear layer in {type(module).__name__}")

    def apply_preferred_action_bias(
            self,
            preferred_action: Sequence[int],
            *,
            gain: float = 1.2,
            clear_existing: bool = True,
            ) -> None:
        """Bias every categorical head toward a concrete action vector.

        BLB Stage-2 has a very large MultiDiscrete action space.  A uniform cold
        start almost never samples the safe all-max baseline, so the first PPO
        rollouts can collapse into identical hard penalties.  This method keeps
        the architecture unchanged while making the initial policy start near a
        caller-provided action, usually ``make_all_max_action_vector``.
        """
        arr = np.asarray(preferred_action, dtype=int).reshape(-1)
        per_layer_width = len(self.per_layer_dims)
        expected = int(self.num_layers) * per_layer_width + 1
        if arr.size != expected:
            raise ValueError(
                f"preferred_action length {arr.size} != expected {expected}"
            )

        layer_linear = self._last_linear(self.layer_head)
        with torch.no_grad():
            if layer_linear.bias is None:
                return
            offsets = np.cumsum([0] + self.per_layer_dims[:-1]).astype(int)
            for dim_idx, (offset, dim) in enumerate(zip(offsets, self.per_layer_dims)):
                per_layer_values = [
                    int(arr[layer_idx * per_layer_width + dim_idx])
                    for layer_idx in range(int(self.num_layers))
                ]
                values, counts = np.unique(per_layer_values, return_counts=True)
                preferred_idx = int(values[int(np.argmax(counts))])
                if preferred_idx < 0 or preferred_idx >= int(dim):
                    raise ValueError(
                        f"preferred action index {preferred_idx} out of range for dim {dim}"
                    )
                start = int(offset)
                end = start + int(dim)
                if clear_existing:
                    layer_linear.bias[start:end].zero_()
                layer_linear.bias[start + preferred_idx] += float(gain)

    def _layer_indices(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.arange(self.num_layers, device=device).unsqueeze(0).expand(batch, -1)

    def forward(self, state: torch.Tensor) -> PolicyForwardResult:
        """``state``: shape [batch, state_dim] → ``PolicyForwardResult``。"""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        batch = state.shape[0]
        h = self.encoder(state)                                    # [B, d_hidden]
        first_input_logits = h.new_zeros((batch, self.first_input_levels))
        value = self.value_head(h).squeeze(-1)                     # [B]

        layer_idx = self._layer_indices(batch, state.device)       # [B, L]
        emb = self.layer_emb(layer_idx)                            # [B, L, d_emb]
        h_per_layer = h.unsqueeze(1).expand(-1, self.num_layers, -1)  # [B, L, d_hidden]
        x = torch.cat([h_per_layer, emb], dim=-1)                  # [B, L, d_hidden + d_emb]
        layer_logits = self.layer_head(x)                          # [B, L, sum(per_layer_dims)]
        layer_logits_flat = layer_logits.reshape(batch, -1)        # [B, L * sum]

        return PolicyForwardResult(
            layer_logits_flat=layer_logits_flat,
            first_input_logits=first_input_logits,
            value=value,
        )

    # ------------------------------------------------------------------
    # 动作采样 / log_prob 计算
    # ------------------------------------------------------------------
    def _split_layer_logits(self, layer_logits_flat: torch.Tensor) -> List[List[torch.Tensor]]:
        """``[B, L*sum] → [[batch_per_dim_layer0], [batch_per_dim_layer1], ...]``。"""
        batch, total = layer_logits_flat.shape
        per_layer_dim_sum = sum(self.per_layer_dims)
        layer_logits = layer_logits_flat.reshape(batch, self.num_layers, per_layer_dim_sum)
        out: List[List[torch.Tensor]] = []
        for li in range(self.num_layers):
            li_logits = layer_logits[:, li, :]                     # [B, sum]
            split = _split_logits_per_dim(li_logits, self.per_layer_dims)
            out.append(split)
        return out

    @staticmethod
    def _fixed_first_input_action(batch: int, device: torch.device) -> torch.Tensor:
        """Deprecated first_input slot: fixed compatibility placeholder."""
        return torch.zeros(int(batch), dtype=torch.long, device=device)

    def sample_action(
            self,
            state: torch.Tensor,
            deterministic: bool = False,
            action_mask: Optional[Sequence[Sequence[bool]]] = None,
            action_bias: Optional[Sequence[Sequence[float]]] = None,
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样一个动作向量（与 spec §4.3 ``MultiDiscrete`` 对齐）。

        Returns:
            (actions[B, total_dim], log_probs[B], value[B])
        其中 total_dim == L * sum(per_layer_dims) + first_input_levels-axis.
        """
        out = self.forward(state)
        per_layer_split = self._split_layer_logits(out.layer_logits_flat)

        actions: List[torch.Tensor] = []     # 收集每个分量的 [B] 整数 tensor
        batch = int(out.value.shape[0])
        log_prob_total = torch.zeros(batch, device=state.device)

        # per-layer per-dim categorical
        cursor = 0
        for layer_split in per_layer_split:
            for dim_logits in layer_split:
                masked_logits = _adjust_logits_for_slot(
                    dim_logits, action_mask, action_bias, cursor,
                )
                dist = torch.distributions.Categorical(logits=masked_logits)
                if deterministic:
                    action = torch.argmax(masked_logits, dim=-1)
                else:
                    action = dist.sample()
                log_prob_total = log_prob_total + dist.log_prob(action)
                actions.append(action)
                cursor += 1

        # first_input is deprecated and never sampled. Keep a fixed tail column
        # only so legacy full action vectors retain their width.
        actions.append(self._fixed_first_input_action(batch, state.device))

        action_vec = torch.stack(actions, dim=-1)     # [B, total_dim]
        return action_vec, log_prob_total, out.value

    def per_dim_entropy(
            self,
            state: torch.Tensor,
            action_mask: Optional[Sequence[Sequence[bool]]] = None,
            action_bias: Optional[Sequence[Sequence[float]]] = None,
            ) -> torch.Tensor:
        """Return per-action-slot entropy averaged over the input batch.

        Output shape: ``[total_dim]`` aligned with the action vector layout:
        ``[layer0_slot0, ..., layer0_slotK, layer1_slot0, ..., first_input]``.

        The aggregate ``entropy`` reported by ``ppo_update`` hides whether
        specific slot kinds (F/W/M/S/R/K) are collapsing early; this helper
        lets the runner break down entropy by kind / block to surface that.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        out = self.forward(state)
        per_layer_split = self._split_layer_logits(out.layer_logits_flat)

        entropies: List[torch.Tensor] = []
        cursor = 0
        for layer_split in per_layer_split:
            for dim_logits in layer_split:
                logits = _adjust_logits_for_slot(
                    dim_logits, action_mask, action_bias, cursor,
                )
                dist = torch.distributions.Categorical(logits=logits)
                entropies.append(dist.entropy().mean(dim=0))   # scalar over batch
                cursor += 1
        # Deprecated first_input placeholder has no policy support and no entropy.
        entropies.append(torch.zeros((), device=state.device))
        return torch.stack(entropies)

    def evaluate_action(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
            action_mask: Optional[Sequence[Sequence[bool]]] = None,
            action_bias: Optional[Sequence[Sequence[float]]] = None,
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """给定 (state, action) → (log_prob, entropy, value)，用于 PPO update。

        ``action`` 应是 ``[B, total_dim]`` 的 long tensor，与 ``sample_action`` 输出相同布局。
        """
        if action.dtype != torch.long:
            action = action.long()
        out = self.forward(state)
        per_layer_split = self._split_layer_logits(out.layer_logits_flat)

        log_prob_total = torch.zeros(state.shape[0], device=state.device)
        entropy_total = torch.zeros(state.shape[0], device=state.device)

        cursor = 0
        for layer_split in per_layer_split:
            for dim_logits in layer_split:
                a_col = action[:, cursor]                           # [B]
                masked_logits = _adjust_logits_for_slot(
                    dim_logits, action_mask, action_bias, cursor,
                )
                dist = torch.distributions.Categorical(logits=masked_logits)
                log_prob_total = log_prob_total + dist.log_prob(a_col)
                entropy_total = entropy_total + dist.entropy()
                cursor += 1

        # first_input is a fixed compatibility tail, not a sampled PPO variable.
        # Ignore any legacy non-zero value here; action_vector_to_cfgs also
        # returns first_input_sf=0 and model installation never consumes it.
        cursor += 1

        if cursor != action.shape[1]:
            raise RuntimeError(
                f"action width mismatch: cursor={cursor}, action width={action.shape[1]}"
            )

        return log_prob_total, entropy_total, out.value


# ---------------------------------------------------------------------------
# Rollout buffer + PPO 更新
# ---------------------------------------------------------------------------
@dataclass
class RolloutSample:
    state: np.ndarray
    action: np.ndarray
    log_prob: float
    reward: float
    value: float


class RolloutBuffer:
    """单步 episode（horizon=1）的轻量级 rollout buffer。

    每条记录就是一个 (state, action, log_prob, reward, value) 五元组。
    PPO 更新前一次性把所有记录组装成 tensor。
    """

    def __init__(self):
        self._samples: List[RolloutSample] = []

    def __len__(self) -> int:
        return len(self._samples)

    def add(
            self,
            state: np.ndarray,
            action: np.ndarray,
            log_prob: float,
            reward: float,
            value: float,
            ) -> None:
        self._samples.append(RolloutSample(
            state=np.asarray(state, dtype=np.float32),
            action=np.asarray(action, dtype=np.int64),
            log_prob=float(log_prob),
            reward=float(reward),
            value=float(value),
        ))

    def clear(self) -> None:
        self._samples.clear()

    def to_tensors(
            self,
            device: torch.device,
            advantage_normalize: bool = True,
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (states, actions, old_log_probs, returns, advantages)。

        horizon=1 时 GAE 退化为 ``advantage = reward - value``，``return = reward``。
        """
        if not self._samples:
            raise RuntimeError("RolloutBuffer is empty")
        states = np.stack([s.state for s in self._samples])
        actions = np.stack([s.action for s in self._samples])
        log_probs = np.array([s.log_prob for s in self._samples], dtype=np.float32)
        rewards = np.array([s.reward for s in self._samples], dtype=np.float32)
        values = np.array([s.value for s in self._samples], dtype=np.float32)

        returns = rewards.copy()
        advantages = returns - values
        if advantage_normalize and advantages.size > 1:
            adv_std = float(advantages.std())
            # When a rollout collapses to a single reward bucket (e.g. nearly
            # every action invalid → all rewards == -invalid_penalty), the
            # critic also learns to predict that constant, so advantages all
            # cluster near zero. Centering and dividing by a tiny std then
            # either zeros out the few valid-action signals or blows them up
            # to numerical noise. Skip normalization in that regime so the
            # rare valid candidates keep an unambiguous gradient direction.
            if adv_std > 1e-3:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        return (
            torch.from_numpy(states).to(device),
            torch.from_numpy(actions).to(device),
            torch.from_numpy(log_probs).to(device),
            torch.from_numpy(returns).to(device),
            torch.from_numpy(advantages).to(device),
        )


@dataclass
class PPOConfig:
    """PPO 训练超参（spec §7.3 起步建议）。"""
    lr: float = 5e-5
    clip_range: float = 0.2
    n_epochs: int = 4
    minibatch_size: int = 64
    ent_coef: float = 0.02
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


def ppo_update(
        policy: BLBStage2Policy,
        optimizer: torch.optim.Optimizer,
        buffer: RolloutBuffer,
        cfg: PPOConfig,
        device: torch.device,
        action_mask: Optional[Sequence[Sequence[bool]]] = None,
        action_bias: Optional[Sequence[Sequence[float]]] = None,
        ) -> dict:
    """对 buffer 做一次 PPO update（n_epochs × minibatch）。

    Returns:
        ``{"policy_loss": ..., "value_loss": ..., "entropy": ..., "clip_fraction": ..., "n_samples": ...}``
    """
    if len(buffer) == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                "clip_fraction": 0.0, "n_samples": 0}

    states, actions, old_log_probs, returns, advantages = buffer.to_tensors(device)
    n = states.shape[0]
    indices = np.arange(n)

    metrics_sum = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                   "clip_fraction": 0.0, "n_minibatches": 0}

    for _ in range(int(cfg.n_epochs)):
        np.random.shuffle(indices)
        mb_size = max(1, int(cfg.minibatch_size))
        for start in range(0, n, mb_size):
            end = min(n, start + mb_size)
            mb_idx = indices[start:end]
            mb_idx_t = torch.from_numpy(mb_idx).long().to(device)
            mb_states = states.index_select(0, mb_idx_t)
            mb_actions = actions.index_select(0, mb_idx_t)
            mb_old_log_probs = old_log_probs.index_select(0, mb_idx_t)
            mb_returns = returns.index_select(0, mb_idx_t)
            mb_adv = advantages.index_select(0, mb_idx_t)

            new_log_probs, entropy, value = policy.evaluate_action(
                mb_states,
                mb_actions,
                action_mask=action_mask,
                action_bias=action_bias,
            )
            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            unclipped = ratio * mb_adv
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * mb_adv
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(value, mb_returns)
            entropy_mean = entropy.mean()

            loss = (
                policy_loss
                + cfg.value_coef * value_loss
                - cfg.ent_coef * entropy_mean
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.max_grad_norm is not None and cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                clip_frac = ((torch.abs(ratio - 1.0) > cfg.clip_range).float()).mean().item()
            metrics_sum["policy_loss"] += float(policy_loss.item())
            metrics_sum["value_loss"] += float(value_loss.item())
            metrics_sum["entropy"] += float(entropy_mean.item())
            metrics_sum["clip_fraction"] += float(clip_frac)
            metrics_sum["n_minibatches"] += 1

    n_mb = max(1, metrics_sum["n_minibatches"])
    out = {
        "policy_loss": metrics_sum["policy_loss"] / n_mb,
        "value_loss": metrics_sum["value_loss"] / n_mb,
        "entropy": metrics_sum["entropy"] / n_mb,
        "clip_fraction": metrics_sum["clip_fraction"] / n_mb,
        "n_samples": int(n),
    }
    return out
