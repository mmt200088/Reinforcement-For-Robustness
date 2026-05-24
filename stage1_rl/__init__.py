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
"""
from __future__ import annotations

from .parallel_runner import (
    EpisodeRollout,
    Stage1ParallelRunner,
    Stage1ParallelRunnerDiagnostics,
    Stage1RolloutWorker,
    build_stage1_parallel_runner,
    derive_episode_seed,
    derive_worker_seed,
    format_diagnostics_line,
    parse_device_ids,
)

__all__ = [
    "EpisodeRollout",
    "Stage1ParallelRunner",
    "Stage1ParallelRunnerDiagnostics",
    "Stage1RolloutWorker",
    "build_stage1_parallel_runner",
    "derive_episode_seed",
    "derive_worker_seed",
    "format_diagnostics_line",
    "parse_device_ids",
]
