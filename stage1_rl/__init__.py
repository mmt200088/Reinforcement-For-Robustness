"""Stage-1 RL multi-GPU rollout package.

The Stage-1 search (GTrXL PPO over GELU / Softmax polynomial degrees per
layer) runs ``stage1_search_episodes`` episodes per training run. Each
episode is a sequence of ``total_layers`` per-layer policy decisions
followed by one BERT forward pass over the proxy split for the final reward.
The BERT forward dominates wall-clock; ``Stage1ParallelRunner`` fans that
work across N GPUs so the same PPO update window (default 120 episodes)
is collected ~N times faster.

Public API is intentionally small; the launcher passes a comma-separated
device list via ``--stage1-rl-devices``, the runner is built once before
the episode loop, and the rest of the training loop in
``layer_importance_evaluator.py`` only sees ``runner.run_window(...)``.

The rollout implementation is imported lazily so torch-free Stage-1 search
helpers remain usable by artifact-only resume paths.
"""
from __future__ import annotations

__all__ = [
    "EpisodeRollout",
    "Stage1ParallelRunner",
    "Stage1ParallelRunnerDiagnostics",
    "Stage1RolloutWorker",
    "assign_global_episodes",
    "build_stage1_parallel_runner",
    "derive_episode_seed",
    "format_diagnostics_line",
    "parse_device_ids",
]

_EXPORT_NAMES = frozenset(__all__)


def __getattr__(name: str):
    if name not in _EXPORT_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import parallel_runner

    value = getattr(parallel_runner, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _EXPORT_NAMES)
