"""Sequential PPO actor-critic + rollout buffer for the per-block env.

Three pieces:
  1. :class:`BLBStage2SequentialPolicy` -- shared trunk + a single
     ``MultiDiscrete`` head sized to ``step_schedule_max_dim`` (13 for L=12).
     At each step the env tells the policy which slots are *active* (a
     boolean mask of length max_step_dim, padding suppressed via -inf logits)
     and what the per-slot ``num_levels`` are (so logits beyond that get
     masked out as well).

  2. :class:`SequentialRolloutBuffer` -- stores per-step
     (state, action, log_prob, value, reward, done, slot_mask) tuples
     organised by episode. ``compute_gae(...)`` produces λ-returns and
     advantages over the horizon.

  3. :func:`sequential_ppo_update` -- standard PPO-clip update over the
     buffered transitions. Aware of variable per-step action width via
     the slot_mask.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

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


class BLBStage2SequentialPolicy(nn.Module):
    """Actor + critic over per-step decisions.

    Forward signature:
        ``state``: ``[B, state_dim]``
        Returns ``logits[B, max_step_dim, max_num_levels]`` and ``value[B]``.

    Sampling and log-prob evaluation accept an additional ``slot_mask`` and
    ``per_slot_num_levels`` so the same head can serve every step type.
    """

    def __init__(self, cfg: SequentialPolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.d_hidden),
            nn.ReLU(),
            nn.Linear(cfg.d_hidden, cfg.d_hidden),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(
            cfg.d_hidden, cfg.max_step_dim * cfg.max_num_levels
        )
        self.value_head = nn.Linear(cfg.d_hidden, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        # Encoder + value head: orthogonal sqrt(2) — standard actor-critic init.
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=float(np.sqrt(2)))
                nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)
        # Action head: gain=0.01 (legacy noise_rl_module_v2 trick — see line ~1066
        # there). Keeps ``W_action @ h`` near zero at init AND after the encoder
        # gets perturbed by value-loss gradient, so the warmstart bias remains
        # the dominant signal in the action distribution for many PPO updates.
        # Without this, the default Kaiming init makes ``|W @ h|`` ~ 4-9 at init,
        # which overwhelms the +3.5 warmstart bias and lets the policy drift
        # off baseline on the first sample episode (observed 2026-05-19).
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        nn.init.constant_(self.action_head.bias, 0.0)

    # ------------------------------------------------------------------
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        h = self.encoder(state)
        logits_flat = self.action_head(h)  # [B, S*L]
        logits = logits_flat.view(
            -1, self.cfg.max_step_dim, self.cfg.max_num_levels
        )
        value = self.value_head(h).squeeze(-1)
        return logits, value

    # ------------------------------------------------------------------
    @staticmethod
    def _build_logit_mask(
            slot_mask: torch.Tensor,
            per_slot_num_levels: torch.Tensor,
            max_num_levels: int,
            ) -> torch.Tensor:
        """Return additive -inf mask for invalid (slot, level) cells.

        ``slot_mask`` is ``[B, max_step_dim]`` boolean (True = active).
        ``per_slot_num_levels`` is ``[B, max_step_dim]`` int (0 for padding).
        Output has shape ``[B, max_step_dim, max_num_levels]``.
        """
        B, S = slot_mask.shape
        levels_idx = torch.arange(max_num_levels, device=slot_mask.device).view(1, 1, -1).expand(B, S, -1)
        # padding-slot rows are entirely -inf
        slot_alive = slot_mask.unsqueeze(-1).expand(-1, -1, max_num_levels)
        # within an active slot, levels >= num_levels[slot] get -inf
        level_valid = levels_idx < per_slot_num_levels.unsqueeze(-1)
        valid = slot_alive & level_valid
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
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample one per-step action.

        Returns:
            actions:  ``[B, max_step_dim]`` long (padding entries are 0)
            log_prob: ``[B]`` summed across active slots
            value:    ``[B]``
        """
        logits, value = self.forward(state)
        logits = logits + self._build_logit_mask(
            slot_mask, per_slot_num_levels, self.cfg.max_num_levels
        )
        # collapse padding rows by setting them to a single dummy distribution
        # so torch.distributions doesn't NaN. We then mask-out their log_prob
        # contribution at the end.
        safe_logits = torch.where(
            torch.isfinite(logits).any(dim=-1, keepdim=True),
            logits,
            torch.zeros_like(logits),
        )
        dist = torch.distributions.Categorical(logits=safe_logits)
        if deterministic:
            actions = torch.argmax(safe_logits, dim=-1)
        else:
            actions = dist.sample()
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
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate (log_prob, entropy, value) for a given action under the
        current policy. Used by PPO update.
        """
        logits, value = self.forward(state)
        logits = logits + self._build_logit_mask(
            slot_mask, per_slot_num_levels, self.cfg.max_num_levels
        )
        safe_logits = torch.where(
            torch.isfinite(logits).any(dim=-1, keepdim=True),
            logits,
            torch.zeros_like(logits),
        )
        dist = torch.distributions.Categorical(logits=safe_logits)
        actions_long = actions.long()
        log_prob_per_slot = dist.log_prob(actions_long) * slot_mask.float()
        log_prob = log_prob_per_slot.sum(dim=-1)
        entropy_per_slot = dist.entropy() * slot_mask.float()
        entropy = entropy_per_slot.sum(dim=-1)
        return log_prob, entropy, value

    def apply_preferred_per_step_bias(
            self,
            preferred_per_slot_idx: Sequence[int],
            *,
            gain: float = 1.5,
            clear_existing: bool = True,
            ) -> None:
        """Bias the action head toward a preferred per-slot index for ALL slots.

        Useful as a warmstart toward the all-max baseline (every slot at its
        max-SF index). For BLB Stage-2 the all-max action picks the *largest*
        index for every slot; passing ``[max_idx_for_kind] * max_step_dim``
        biases the policy toward that. Only the diagonal entries are set;
        non-active slots are still masked out at sample time.
        """
        if len(preferred_per_slot_idx) != self.cfg.max_step_dim:
            raise ValueError(
                f"preferred length {len(preferred_per_slot_idx)} != max_step_dim {self.cfg.max_step_dim}"
            )
        with torch.no_grad():
            if self.action_head.bias is None:
                return
            bias = self.action_head.bias.view(self.cfg.max_step_dim, self.cfg.max_num_levels)
            for slot_idx, lvl in enumerate(preferred_per_slot_idx):
                lvl = int(lvl)
                if lvl < 0 or lvl >= self.cfg.max_num_levels:
                    raise ValueError(
                        f"preferred index {lvl} out of range [0, {self.cfg.max_num_levels})"
                    )
                if clear_existing:
                    bias[slot_idx].zero_()
                bias[slot_idx, lvl] += float(gain)


# ---------------------------------------------------------------------------
# Rollout buffer + GAE
# ---------------------------------------------------------------------------

@dataclass
class SequentialTransition:
    state: np.ndarray             # [state_dim]
    action: np.ndarray            # [max_step_dim]
    slot_mask: np.ndarray         # [max_step_dim] bool
    per_slot_num_levels: np.ndarray  # [max_step_dim] int
    log_prob: float
    value: float
    reward: float
    done: bool


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
            log_prob: float,
            value: float,
            reward: float,
            done: bool,
            ) -> None:
        self._buf.append(SequentialTransition(
            state=np.asarray(state, dtype=np.float32),
            action=np.asarray(action, dtype=np.int64),
            slot_mask=np.asarray(slot_mask, dtype=bool),
            per_slot_num_levels=np.asarray(per_slot_num_levels, dtype=np.int64),
            log_prob=float(log_prob),
            value=float(value),
            reward=float(reward),
            done=bool(done),
        ))

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
        rewards = np.array([t.reward for t in self._buf], dtype=np.float32)
        values = np.array([t.value for t in self._buf], dtype=np.float32)
        dones = np.array([t.done for t in self._buf], dtype=bool)

        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in range(n - 1, -1, -1):
            if dones[t] or t == n - 1:
                next_value = 0.0
                next_nonterm = 0.0
            else:
                next_value = float(values[t + 1])
                next_nonterm = 1.0 - float(dones[t + 1])
            delta = float(rewards[t]) + gamma * next_value * (0.0 if dones[t] else 1.0) - float(values[t])
            last_gae = delta + gamma * lam * (0.0 if dones[t] else 1.0) * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return returns, advantages

    def to_tensors(
            self,
            device: torch.device,
            *,
            gamma: float = 0.99,
            lam: float = 0.95,
            advantage_normalize: bool = True,
            ):
        """Pack everything into batched tensors. Returns:
            states, actions, slot_masks, per_slot_num_levels,
            old_log_probs, returns, advantages
        """
        if not self._buf:
            raise RuntimeError("SequentialRolloutBuffer is empty")
        states = np.stack([t.state for t in self._buf])
        actions = np.stack([t.action for t in self._buf])
        slot_masks = np.stack([t.slot_mask for t in self._buf])
        levels = np.stack([t.per_slot_num_levels for t in self._buf])
        log_probs = np.array([t.log_prob for t in self._buf], dtype=np.float32)
        returns, advantages = self.compute_gae(gamma=gamma, lam=lam)
        if advantage_normalize and advantages.size > 1:
            adv_std = float(advantages.std())
            if adv_std > 1e-3:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
        return (
            torch.from_numpy(states).to(device),
            torch.from_numpy(actions).to(device),
            torch.from_numpy(slot_masks).to(device),
            torch.from_numpy(levels).to(device),
            torch.from_numpy(log_probs).to(device),
            torch.from_numpy(returns).to(device),
            torch.from_numpy(advantages).to(device),
        )


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

@dataclass
class SequentialPPOConfig:
    lr: float = 3e-4
    clip_range: float = 0.2
    n_epochs: int = 4
    minibatch_size: int = 128
    ent_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    gamma: float = 0.99
    gae_lambda: float = 0.95


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
                "clip_fraction": 0.0, "n_samples": 0, "ent_coef": 0.0}
    effective_ent_coef = (
        float(cfg.ent_coef) if ent_coef_override is None else float(ent_coef_override)
    )

    states, actions, slot_masks, levels, old_log_probs, returns, advantages = buffer.to_tensors(
        device, gamma=cfg.gamma, lam=cfg.gae_lambda,
    )
    n = states.shape[0]
    indices = np.arange(n)

    metrics_sum = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                   "clip_fraction": 0.0, "n_minibatches": 0}

    for _ in range(int(cfg.n_epochs)):
        np.random.shuffle(indices)
        mb_size = max(1, int(cfg.minibatch_size))
        for start in range(0, n, mb_size):
            end = min(n, start + mb_size)
            mb = torch.from_numpy(indices[start:end]).long().to(device)
            new_log_probs, entropy, value = policy.evaluate_action(
                states.index_select(0, mb),
                actions.index_select(0, mb),
                slot_masks.index_select(0, mb),
                levels.index_select(0, mb),
            )
            old_lp = old_log_probs.index_select(0, mb)
            ret = returns.index_select(0, mb)
            adv = advantages.index_select(0, mb)
            ratio = torch.exp(new_log_probs - old_lp)
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * adv
            policy_loss = -torch.min(unclipped, clipped).mean()
            # Huber (delta=1.0) instead of MSE: caps value gradient magnitude
            # at delta, so unnormalised returns (~+37) don't blow up the
            # shared-encoder gradient and overwhelm the policy gradient
            # (observed 2026-05-19: value_loss ~ 60.86 vs policy_loss ~ -0.054
            # → 1126x ratio caused encoder to drift off warmstart-bias
            # configuration). Matches legacy noise_rl_module_v2 (line ~1886).
            value_loss = F.huber_loss(value, ret, delta=1.0)
            entropy_mean = entropy.mean()
            loss = (
                policy_loss
                + cfg.value_coef * value_loss
                - effective_ent_coef * entropy_mean
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
    return {
        "policy_loss": metrics_sum["policy_loss"] / n_mb,
        "value_loss": metrics_sum["value_loss"] / n_mb,
        "entropy": metrics_sum["entropy"] / n_mb,
        "clip_fraction": metrics_sum["clip_fraction"] / n_mb,
        "n_samples": int(n),
        "ent_coef": float(effective_ent_coef),
    }


# ---------------------------------------------------------------------------
# Convenience: derive (slot_mask, per_slot_num_levels) for one BlockStepSpec
# ---------------------------------------------------------------------------

def step_to_mask_and_levels(
        spec,        # BlockStepSpec from action_space
        max_step_dim: int,
        max_num_levels: int,
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (slot_mask[max_step_dim] bool, per_slot_num_levels[max_step_dim] int).

    Active slots = slot_dims of this spec, padded to max_step_dim with zeros.
    Per-slot num_levels = each slot's num_levels (clamped to max_num_levels).
    """
    n = len(spec.slot_dims)
    if n > max_step_dim:
        raise ValueError(f"step has {n} slots > max_step_dim={max_step_dim}")
    slot_mask = np.zeros(max_step_dim, dtype=bool)
    slot_mask[:n] = True
    levels = np.zeros(max_step_dim, dtype=np.int64)
    for i, d in enumerate(spec.slot_dims):
        if int(d) > max_num_levels:
            raise ValueError(
                f"slot {i} has {d} levels > max_num_levels={max_num_levels}"
            )
        levels[i] = int(d)
    return slot_mask, levels
