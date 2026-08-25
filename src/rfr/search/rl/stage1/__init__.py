"""Stage-1 PPO rollout and checkpoint runtime."""

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


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import parallel_runner

    value = getattr(parallel_runner, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
