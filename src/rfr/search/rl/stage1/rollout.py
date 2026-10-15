"""Stage-1 recurrent rollout storage and tensor packing."""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import torch
from rfr.search.rl.stage1 import config as stage1_config

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
    mask_key = tuple(bool(x) for x in stage1_config.STAGE1_GELU_ACTION_MASK.tolist())
    key = (str(device), mask_key)
    cached = _stage1_gelu_mask_template_cache.get(key)
    if cached is not None:
        return cached

    mask_np = np.asarray(stage1_config.STAGE1_GELU_ACTION_MASK, dtype=bool).copy()
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
