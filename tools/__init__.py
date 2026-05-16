"""tools — project-level utilities used by the launcher, multi-seed driver,
experiment tracking, and paper figure pipeline.

Each module here is importable AND runnable as a CLI::

    python -m tools.experiments_log query --dataset mrpc
    python tools/experiments_log.py query --dataset mrpc          # same thing

Modules:
  - ``experiments_log``    — append-only experiment registry + index rebuild
  - ``aggregate_seeds``    — multi-seed run aggregator (mean ± std)
  - ``paper_figures``      — paper-friendly figure generator
  - ``validate_preset``    — preset .conf typo / unknown-flag detector
  - ``status_board``       — running RL/GA/general/compare job aggregator (legacy)

The ``tools/run_multi_seed.sh`` shell driver lives alongside but is not
importable (it shells out to the launcher with --blb-v3-seed / --run-tag).
"""
from __future__ import annotations

__all__ = [
    "aggregate_seeds",
    "experiments_log",
    "paper_figures",
    "validate_preset",
]
