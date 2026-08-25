"""Atomic checkpoint ownership for the production Stage-1 policy."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from rfr.preparation.data.protocol import PROTOCOL_SCHEMA, validate_dataset_protocol_binding


STAGE1_CHECKPOINT_FILENAME = "stage1_rl_checkpoint.pt"


def _serialize_numpy(obj):
    """Recursively convert NumPy arrays to lists for checkpoint serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _serialize_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_serialize_numpy(item) for item in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted
    return obj


def _deserialize_numpy_keys(obj, keys_to_convert):
    """Restore selected list values as integer NumPy arrays."""
    if not isinstance(obj, dict):
        return obj
    for key in keys_to_convert:
        if key in obj and obj[key] is not None:
            if isinstance(obj[key], list):
                obj[key] = np.array(obj[key], dtype=int)
    return obj


def _atomic_torch_save(obj, path):
    """Commit a checkpoint only after the temporary file is complete."""
    tmp_path = path + ".tmp"
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:

        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


_STAGE1_DETAIL_PREFIX = "ppo_step_info_"
_STAGE1_DETAIL_SUFFIX = ".txt"


def _is_stage1_detail_filename(name):
    return (
        name.startswith(_STAGE1_DETAIL_PREFIX)
        and name.endswith(_STAGE1_DETAIL_SUFFIX)
    )


def stage1_detail_file_sizes(details_dir):
    """Return append boundaries for Stage-1 detail chunks."""
    directory = os.fspath(details_dir)
    if not os.path.isdir(directory):
        return {}
    sizes = {}
    with os.scandir(directory) as entries:
        for entry in entries:
            if (
                _is_stage1_detail_filename(entry.name)
                and entry.is_file(follow_symlinks=False)
            ):
                sizes[entry.name] = int(entry.stat(follow_symlinks=False).st_size)
    return dict(sorted(sizes.items()))


def recover_stage1_detail_files(details_dir, committed_sizes):
    """Roll Stage-1 detail chunks back to one checkpoint transaction."""
    if committed_sizes is None:
        return
    directory = os.fspath(details_dir)
    os.makedirs(directory, exist_ok=True)
    normalized = {}
    for raw_name, raw_size in dict(committed_sizes).items():
        name = str(raw_name)
        if os.path.basename(name) != name or not _is_stage1_detail_filename(name):
            raise ValueError(f"invalid Stage-1 detail checkpoint name: {name!r}")
        size = int(raw_size)
        if size < 0:
            raise ValueError(
                f"invalid Stage-1 detail checkpoint size for {name!r}: {size}"
            )
        normalized[name] = size

    with os.scandir(directory) as entries:
        existing = {
            entry.name
            for entry in entries
            if (
                _is_stage1_detail_filename(entry.name)
                and entry.is_file(follow_symlinks=False)
            )
        }
    for name in existing - normalized.keys():
        os.remove(os.path.join(directory, name))

    for name, committed_size in normalized.items():
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            raise RuntimeError(
                f"Stage-1 detail file missing at checkpoint recovery: {path}"
            )
        current_size = os.path.getsize(path)
        if current_size < committed_size:
            raise RuntimeError(
                f"Stage-1 detail file is shorter than checkpoint boundary: "
                f"{path} ({current_size} < {committed_size})"
            )
        if current_size > committed_size:
            with open(path, "r+b") as handle:
                handle.truncate(committed_size)


STAGE1_CUDA_RNG_ROLE_REGISTRY_VERSION = 1


def merge_stage1_cuda_rng_role_registry(
    previous_registry,
    active_role_states,
):
    """Update visible logical roles while retaining temporarily absent roles."""
    registry = list(previous_registry or ())
    for role_index, state in enumerate(active_role_states):
        if role_index < len(registry):
            registry[role_index] = state
        else:
            registry.append(state)
    return registry


def resolve_stage1_cuda_rng_role_registry(
    checkpoint,
    *,
    active_role_count,
    new_role_state_factory,
):
    """Resolve Stage-1 CUDA RNG streams independently of physical GPU IDs."""
    current_count = int(active_role_count)
    if current_count < 0:
        raise ValueError("active CUDA RNG role count must be non-negative")

    stored_registry = checkpoint.get("cuda_rng_state_by_role")
    if stored_registry is None:
        raise RuntimeError(
            "Stage-1 checkpoint is missing the CUDA RNG role registry; "
            "a fresh run is required"
        )

    version = int(
        checkpoint.get("cuda_rng_role_registry_version", 0) or 0
    )
    if version != STAGE1_CUDA_RNG_ROLE_REGISTRY_VERSION:
        raise RuntimeError(
            "unsupported Stage-1 checkpoint CUDA RNG role registry "
            f"version: {version}"
        )
    registry = list(stored_registry)
    saved_active_count = int(
        checkpoint.get("cuda_rng_active_role_count", len(registry))
    )
    if saved_active_count < 0 or saved_active_count > len(registry):
        raise RuntimeError(
            "Stage-1 checkpoint CUDA RNG active role count is invalid"
        )
    if current_count == 0 and saved_active_count > 0:
        raise RuntimeError(
            "Stage-1 checkpoint requires CUDA but no healthy GPU is visible"
        )
    if current_count > 0 and saved_active_count == 0:
        raise RuntimeError(
            "Stage-1 checkpoint was created without CUDA; changing the "
            "training backend cannot preserve exact results"
        )
    while len(registry) < current_count:
        registry.append(new_role_state_factory(len(registry)))
    return registry, list(registry[:current_count])


def save_stage1_rl_checkpoint(
    path,
    gtrxl_net,
    optimizer,
    episode,
    gtrxl_ppo_update_count,
    episode_rewards,
    episode_losses,
    episode_metric1s,
    episode_metric2s,
    episode_entropies,
    best_reward,
    best_cost,
    best_config,
    search_best_config,
    global_best_config,
    window_best_reward,
    window_best_cost,
    window_best_config,
    ev_runtime_state,
    stage1_prev_avg_reward,
    stage1_warnings,
    dataset_protocol_hash=None,
    structured_run_id=None,
    structured_jsonl_sizes=None,
    detail_file_sizes=None,
    cuda_rng_role_registry=None,
):
    """Save a complete Stage 1 RL training checkpoint."""
    if not str(dataset_protocol_hash or ""):
        raise ValueError("Stage-1 checkpoint requires dataset_protocol_hash")
    active_cuda_rng_states = (
        [state.cpu() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )
    cuda_rng_role_registry = merge_stage1_cuda_rng_role_registry(
        cuda_rng_role_registry,
        active_cuda_rng_states,
    )
    checkpoint = {
        "version": 2,
        "dataset_protocol_schema": PROTOCOL_SCHEMA,
        "completed_episodes": episode + 1,
        "gtrxl_net_state_dict": gtrxl_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "gtrxl_ppo_update_count": gtrxl_ppo_update_count,
        "episode_rewards": list(episode_rewards),
        "episode_losses": list(episode_losses),
        "episode_metric1s": list(episode_metric1s),
        "episode_metric2s": list(episode_metric2s),
        "episode_entropies": list(episode_entropies),
        "best_reward": best_reward,
        "best_cost": best_cost,
        "best_config": _serialize_numpy(best_config),
        "search_best_config": _serialize_numpy(search_best_config),
        "global_best_config": _serialize_numpy(global_best_config),
        "window_best_reward": window_best_reward,
        "window_best_cost": window_best_cost,
        "window_best_config": _serialize_numpy(window_best_config),
        "ev_runtime_state": ev_runtime_state,
        "stage1_prev_avg_reward": stage1_prev_avg_reward,
        "stage1_warnings": _serialize_numpy(stage1_warnings),
        "dataset_protocol_hash": (
            str(dataset_protocol_hash)
            if dataset_protocol_hash is not None else None
        ),
        "structured_run_id": (
            str(structured_run_id) if structured_run_id is not None else None
        ),
        "structured_jsonl_sizes": (
            {
                str(name): int(size)
                for name, size in dict(structured_jsonl_sizes).items()
            }
            if structured_jsonl_sizes is not None else None
        ),
        "detail_file_sizes": (
            {
                str(name): int(size)
                for name, size in dict(detail_file_sizes).items()
            }
            if detail_file_sizes is not None else None
        ),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": active_cuda_rng_states or None,
        "cuda_rng_role_registry_version": (
            STAGE1_CUDA_RNG_ROLE_REGISTRY_VERSION
        ),
        "cuda_rng_state_by_role": cuda_rng_role_registry,
        "cuda_rng_active_role_count": len(active_cuda_rng_states),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    _atomic_torch_save(checkpoint, path)


def load_stage1_rl_checkpoint(
    path,
    gtrxl_net,
    optimizer,
    device="cuda",
    *,
    expected_dataset_protocol_hash,
):
    """Load a Stage 1 RL checkpoint and return the restored state."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    validate_dataset_protocol_binding(
        checkpoint,
        expected_hash=expected_dataset_protocol_hash,
        artifact="Stage-1 checkpoint",
    )
    gtrxl_net.load_state_dict(checkpoint["gtrxl_net_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    _np_keys = ["gelu", "softmax"]
    for cfg_key in ("best_config", "search_best_config", "global_best_config", "window_best_config"):
        if checkpoint.get(cfg_key) is not None:
            checkpoint[cfg_key] = _deserialize_numpy_keys(
                checkpoint[cfg_key], _np_keys,
            )
    if checkpoint.get("torch_rng_state") is not None:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    active_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    initial_active_states = (
        [state.cpu() for state in torch.cuda.get_rng_state_all()]
        if active_count > 0
        else []
    )
    registry, active_states = resolve_stage1_cuda_rng_role_registry(
        checkpoint,
        active_role_count=active_count,
        new_role_state_factory=lambda role_index: initial_active_states[
            role_index
        ],
    )
    for role_index, state in enumerate(active_states):
        torch.cuda.set_rng_state(state.cpu(), device=role_index)
    checkpoint["cuda_rng_role_registry_version"] = (
        STAGE1_CUDA_RNG_ROLE_REGISTRY_VERSION
    )
    checkpoint["cuda_rng_state_by_role"] = registry
    checkpoint["cuda_rng_active_role_count"] = active_count
    if checkpoint.get("numpy_rng_state") is not None:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if checkpoint.get("python_rng_state") is not None:
        random.setstate(checkpoint["python_rng_state"])
    return checkpoint
